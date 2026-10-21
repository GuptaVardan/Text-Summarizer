import importlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import torch
from transformers import PegasusConfig, PegasusForConditionalGeneration

import app as api
from evaluate_model import percentile, rouge_scores
from src.textSummarizer.components.model_evaluation import ModelEvaluation
from src.textSummarizer.components.model_trainer import ModelTrainer
from src.textSummarizer.components.data_transformation import DataTransformation
from src.textSummarizer.pipeline.predicition_pipeline import PredictionPipeline
from src.textSummarizer.utils.rouge_metrics import RougeMetric


class APIRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.predictor = MagicMock()
        self.predictor.predict.return_value = 'Sam will call Jo.'
        api.app.dependency_overrides[api.predictor_dependency] = lambda: self.predictor
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url='http://test')

    async def asyncTearDown(self):
        await self.client.aclose()
        api.app.dependency_overrides.clear()
        api.get_predictor.cache_clear()

    async def test_json_request_and_response(self):
        response = await self.client.post('/predict', json={'text': 'Sam: I will call you, Jo.'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'summary': 'Sam will call Jo.'})

    async def test_invalid_inputs(self):
        for payload in ({}, {'text': ''}, {'text': '   '}, {'text': 'a' * 20001}):
            response = await self.client.post('/predict', json=payload)
            self.assertEqual(response.status_code, 422)
        self.predictor.predict.assert_not_called()

    async def test_missing_checkpoint_returns_503(self):
        api.app.dependency_overrides.clear()
        with patch.object(api, 'get_predictor', side_effect=FileNotFoundError('missing')):
            response = await self.client.post('/predict', json={'text': 'Sam: hi'})
        self.assertEqual(response.status_code, 503)
        self.assertNotIn('missing', response.text)

    async def test_generation_failure_returns_500(self):
        self.predictor.predict.side_effect = RuntimeError('private details')
        response = await self.client.post('/predict', json={'text': 'Sam: hi'})
        self.assertEqual(response.status_code, 500)
        self.assertNotIn('private details', response.text)

    async def test_training_is_not_an_http_route(self):
        self.assertEqual((await self.client.get('/train')).status_code, 404)


class PipelineRegressionTests(unittest.TestCase):
    def test_predictor_is_cached(self):
        api.get_predictor.cache_clear()
        with patch.object(api, 'PredictionPipeline') as factory:
            self.assertIs(api.get_predictor(), api.get_predictor())
            factory.assert_called_once()
        api.get_predictor.cache_clear()

    def test_missing_checkpoint_fails_without_download(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                PredictionPipeline(Path(directory) / 'missing', directory)

    def test_training_keeps_test_split_held_out(self):
        module = importlib.import_module('src.textSummarizer.components.model_trainer')
        config = SimpleNamespace(root_dir='unused', data_path='unused', model_ckpt='unused',
            num_train_epochs=1, warmup_steps=0, per_device_train_batch_size=1,
            weight_decay=.01, logging_steps=1, evaluation_strategy='steps',
            eval_steps=1, save_steps=1000, gradient_accumulation_steps=16)
        splits = {'train': ['train1', 'train2'], 'validation': ['validation'], 'test': ['test']}
        with patch.object(module, 'AutoTokenizer'), patch.object(module, 'AutoModelForSeq2SeqLM'), \
             patch.object(module, 'DataCollatorForSeq2Seq'), patch.object(module, 'TrainingArguments') as arguments, \
             patch.object(module, 'Trainer') as trainer, patch.object(module, 'load_from_disk', return_value=splits):
            trainer.return_value.train.return_value.metrics = {'train_loss': 1.0}
            ModelTrainer(config).train()
            trainer.return_value.train.assert_called_once_with(resume_from_checkpoint=None)
            trainer.return_value.train.reset_mock()
            ModelTrainer(config).train(resume_from_checkpoint='checkpoint-250')
            trainer.return_value.train.assert_called_once_with(resume_from_checkpoint='checkpoint-250')
        self.assertIs(trainer.call_args.kwargs['train_dataset'], splits['train'])
        self.assertIs(trainer.call_args.kwargs['eval_dataset'], splits['validation'])
        self.assertEqual(arguments.call_args.kwargs['gradient_accumulation_steps'], 16)

    def test_evaluation_preserves_words_and_disables_gradients(self):
        model, tokenizer, metric = MagicMock(), MagicMock(), MagicMock()
        tokenizer.return_value = {'input_ids': torch.tensor([[1, 2]]), 'attention_mask': torch.tensor([[1, 1]])}
        tokenizer.decode.return_value = 'Alice calls Bob.'
        def generate(**kwargs):
            self.assertFalse(torch.is_grad_enabled())
            return [torch.tensor([1, 2])]
        model.generate.side_effect = generate
        ModelEvaluation(None).calculate_metric_on_test_ds(
            {'dialogue': ['Alice: calling Bob'], 'summary': ['Alice calls Bob.']},
            metric, model, tokenizer, batch_size=1, device='cpu',
            column_text='dialogue', column_summary='summary')
        metric.add_batch.assert_called_once_with(predictions=['Alice calls Bob.'], references=['Alice calls Bob.'])
        model.eval.assert_called_once()

    def test_tokenizer_uses_target_api(self):
        component = DataTransformation.__new__(DataTransformation)
        component.tokenizer = MagicMock(side_effect=[{'input_ids': [[1]], 'attention_mask': [[1]]}, {'input_ids': [[2]]}])
        result = component.convert_examples_to_features({'dialogue': ['a'], 'summary': ['b']})
        self.assertEqual(result['labels'], [[2]])
        self.assertEqual(component.tokenizer.call_args.kwargs['text_target'], ['b'])

    def test_invalid_dialogue_summary_pairs_are_rejected(self):
        for value in (None, '', '   ', 42):
            self.assertFalse(DataTransformation.valid_example({'dialogue': value, 'summary': 'valid'}))
            self.assertFalse(DataTransformation.valid_example({'dialogue': 'valid', 'summary': value}))
        self.assertTrue(DataTransformation.valid_example({'dialogue': 'Alice: hi', 'summary': 'Alice greets.'}))

    def test_real_pegasus_forward_and_beam_search(self):
        # Random tiny architecture: validates model API without pretrained downloads.
        config = PegasusConfig(vocab_size=32, d_model=16, encoder_layers=1, decoder_layers=1,
            encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=32,
            decoder_ffn_dim=32, max_position_embeddings=32, pad_token_id=0,
            eos_token_id=1, decoder_start_token_id=0)
        model = PegasusForConditionalGeneration(config)
        inputs = torch.tensor([[3, 4, 1, 0]])
        attention = torch.tensor([[1, 1, 1, 0]])
        loss = model(input_ids=inputs, attention_mask=attention, labels=torch.tensor([[5, 1, -100]])).loss
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        model.eval()
        with torch.inference_mode():
            generated = model.generate(input_ids=inputs, attention_mask=attention, num_beams=2, max_new_tokens=4)
        self.assertEqual(generated.shape[0], 1)

    def test_metric_scale_and_percentiles(self):
        scores = rouge_scores(['Alice calls Bob'], ['Alice calls Bob'])
        self.assertEqual(scores['rouge1'], 100)
        self.assertEqual(scores['rougeL'], 100)
        self.assertEqual(percentile([1, 2, 3], .5), 2)
        with self.assertRaises(ValueError):
            rouge_scores([], [])

    def test_batched_and_standalone_rouge_protocols_match(self):
        predictions = ['Alice calls Bob', 'Jo stays home']
        references = ['Alice calls Bob', 'Jo will stay home']
        metric = RougeMetric()
        for prediction, reference in zip(predictions, references):
            metric.add_batch(predictions=[prediction], references=[reference])
        self.assertEqual(metric.compute(), rouge_scores(predictions, references))

    def test_cleaning_records_counts_and_filters_null_before_tokenizing(self):
        from datasets import Dataset, DatasetDict, load_from_disk
        from src.textSummarizer.components import data_transformation as module
        import json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = DatasetDict({
                'train': Dataset.from_dict({'dialogue': ['Alice: hi', None], 'summary': ['Alice greets.', 'invalid']}),
                'validation': Dataset.from_dict({'dialogue': ['Bob: hi'], 'summary': ['Bob greets.']}),
                'test': Dataset.from_dict({'dialogue': ['Jo: hi'], 'summary': ['Jo greets.']}),
            })
            dataset.save_to_disk(str(root / 'raw'))
            component = DataTransformation.__new__(DataTransformation)
            component.config = SimpleNamespace(data_path=str(root / 'raw'), root_dir=str(root / 'clean'))
            def tokenize(rows):
                self.assertTrue(all(isinstance(text, str) and text.strip() for text in rows['dialogue']))
                return {'input_ids': [[1]] * len(rows['dialogue']),
                        'attention_mask': [[1]] * len(rows['dialogue']),
                        'labels': [[2]] * len(rows['dialogue'])}
            component.convert_examples_to_features = tokenize
            component.convert()
            counts = json.loads((root / 'clean/cleaning_counts.json').read_text())
            self.assertEqual(counts['train'], {'before': 2, 'after': 1, 'removed': 1})
            saved = load_from_disk(str(root / 'clean/samsum_dataset'))
            self.assertEqual(len(saved['train']), 1)
            self.assertEqual(saved['train'][0]['labels'], [2])


if __name__ == '__main__':
    unittest.main()
