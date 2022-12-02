from abc import ABC, abstractmethod
from typing import List, Optional, Dict

import torch
from torch.nn import Module, Embedding
from transformers import BertTokenizer, RobertaTokenizer, BertForMaskedLM, RobertaForMaskedLM, GPT2Tokenizer, DebertaTokenizer, DebertaForMaskedLM, AutoModelForMaskedLM, AutoConfig, AutoModel, AutoTokenizer

import log
from patterns import WORD_TOKEN, MASK_TOKEN

logger = log.get_logger('root')


class OverwriteableEmbedding(Module):

    def __init__(self, embedding: Embedding, overwrite_fct=None):
        super().__init__()
        self.embedding = embedding
        self.overwrite_fct = overwrite_fct

    def forward(self, inp: torch.Tensor):
        embds = self.embedding(inp)
        if self.overwrite_fct is not None:
            embds = self.overwrite_fct(embds)
        return embds


class AbstractMaskedLanguageModel(ABC):
    @abstractmethod
    def get_predictions(self, pattern: str, base_word: str, num_predictions: int) -> List[str]:
        pass


class MockMaskedLanguageModel(AbstractMaskedLanguageModel):
    def get_predictions(self, pattern: str, base_word: str, num_predictions: int) -> List[str]:
        return ['cat', 'dog', 'coffee', 'mouse', 'tree', 'apple', 'orange']


class BertMaskedLanguageModel(AbstractMaskedLanguageModel):
    tokenizer_cls = BertTokenizer
    model_cls = BertForMaskedLM
    model_str = 'bert'

    def __init__(self, model_name: str, embeddings: Optional[Dict[str, torch.Tensor]] = None):
        self.tokenizer = type(self).tokenizer_cls.from_pretrained(model_name)
        self.model = type(self).model_cls.from_pretrained(model_name)
        self.model.eval()

        word_embeddings = getattr(self.model, type(self).model_str).embeddings.word_embeddings
        getattr(self.model, type(self).model_str).embeddings.word_embeddings = OverwriteableEmbedding(word_embeddings)
        self.embeddings = embeddings

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        if self.embeddings:
            for embedding in self.embeddings.values():
                embedding.to(self.device)

    def get_predictions(self, pattern: str, base_word: str, num_predictions: int) -> List[str]:

        replace_base_word = self.embeddings and base_word in self.embeddings

        pattern = pattern.replace(MASK_TOKEN, self.tokenizer.mask_token)
        left_context, right_context = pattern.split(WORD_TOKEN)

        if not replace_base_word:
            model_input = self._prepare_text(''.join([left_context, base_word, right_context]))
            logger.debug('Inferring embedding for {} without replacement'.format(model_input['tokenized_text']))

        else:
            model_input = self._prepare_text(' '.join([left_context, self.tokenizer.unk_token, right_context]))
            base_word_idx = model_input['tokenized_text'].index(self.tokenizer.unk_token)

            if model_input['tokenized_text'][base_word_idx] != self.tokenizer.unk_token:
                raise ValueError("Got wrong base_word_idx, word at position {} is {} and not [UNK]".format(
                    base_word_idx, model_input['tokenized_text'][base_word_idx]))

            getattr(self.model, type(self).model_str).embeddings.word_embeddings.overwrite_fct \
                = lambda embeddings: self._overwrite_embeddings(embeddings, base_word_idx, self.embeddings[base_word])

            logger.debug(
                'Inferring embedding for {} with replacement, base_word_idx = {}'.format(model_input['tokenized_text'],
                                                                                         base_word_idx))

        if len(model_input['masked_indices']) != 1:
            raise ValueError(
                'The pattern must contain exactly one "{}", got "{}" with base word "{}"'.format(
                    self.tokenizer.mask_token, pattern, base_word)
            )

        with torch.no_grad():
            predictions = self.model(
                input_ids=model_input['tokens'].to(self.device),
                token_type_ids=model_input['segments'].to(self.device)
            )[0]

        getattr(self.model, type(self).model_str).embeddings.word_embeddings.overwrite_fct = None
        predicted_tokens = []

        for masked_index in model_input['masked_indices']:
            _, predicted_indices = torch.topk(predictions[0, masked_index], num_predictions)

            for i in range(len(predicted_indices)):
                predicted_index = predicted_indices[i]
                predicted_token = self.tokenizer.convert_ids_to_tokens([predicted_index.item()])[0]
                predicted_tokens.append(predicted_token)
        return predicted_tokens

    def _prepare_text(self, text, use_sep=True, use_cls=True, use_full_stop=True):
        if use_cls:
            text = self.tokenizer.cls_token + ' ' + text
        if use_full_stop and not text[len(text) - 1] in ['?', '.', '!']:
            text += '.'
        if use_sep:
            text += ' ' + self.tokenizer.sep_token

        if isinstance(self.tokenizer, GPT2Tokenizer):
            tokenized_text = self.tokenizer.tokenize(text, add_prefix_space=True)
        else:
            tokenized_text = self.tokenizer.tokenize(text)

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [0] * len(indexed_tokens)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        # get all masked indices
        masked_indices = [i for i, x in enumerate(tokenized_text) if x == self.tokenizer.mask_token]

        return {'tokenized_text': tokenized_text,
                'tokens': tokens_tensor,
                'segments': segments_tensors,
                'masked_indices': masked_indices}

    @staticmethod
    def _overwrite_embeddings(embeddings: torch.Tensor, index: int, replacement_embedding: torch.Tensor):
        # this function is currently not designed to work with more than one batch
        if embeddings.shape[0] != 1:
            raise ValueError('expected a batch of size 1 but found ' + str(embeddings.shape[0].item()))

        embeddings[0, index, :] = replacement_embedding
        return embeddings

#if (args.model_cls == 'bertram'):
    #model = model_cls(args.model_name, 'bert-base-uncased', embeddings)

class CustomMaskedLanguageModel(AbstractMaskedLanguageModel):
    tokenizer_cls = "tmp"
    model_cls = "tmp"
    model_str = 'custom'
    model_path = "tmp"
    config = AutoConfig.from_pretrained('./dataset/outputs/BERT_fused_trained/')

    def __init__(self, model_name: str, token_name: str, embeddings: Optional[Dict[str, torch.Tensor]] = None):
        self.tokenizer = type(self).tokenizer_cls.from_pretrained(token_name)
        self.model = type(self).model_cls.from_config(self.config)
        self.model.eval()

        word_embeddings = getattr(self.model, type(self).model_str).embeddings.word_embeddings  #issue here
        getattr(self.model, type(self).model_str).embeddings.word_embeddings = OverwriteableEmbedding(word_embeddings)
        self.embeddings = embeddings

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        if self.embeddings:
            for embedding in self.embeddings.values():
                embedding.to(self.device)

    def get_predictions(self, pattern: str, base_word: str, num_predictions: int) -> List[str]:

        replace_base_word = self.embeddings and base_word in self.embeddings

        pattern = pattern.replace(MASK_TOKEN, self.tokenizer.mask_token)
        left_context, right_context = pattern.split(WORD_TOKEN)

        if not replace_base_word:
            model_input = self._prepare_text(''.join([left_context, base_word, right_context]))
            logger.debug('Inferring embedding for {} without replacement'.format(model_input['tokenized_text']))

        else:
            model_input = self._prepare_text(' '.join([left_context, self.tokenizer.unk_token, right_context]))
            base_word_idx = model_input['tokenized_text'].index(self.tokenizer.unk_token)

            if model_input['tokenized_text'][base_word_idx] != self.tokenizer.unk_token:
                raise ValueError("Got wrong base_word_idx, word at position {} is {} and not [UNK]".format(
                    base_word_idx, model_input['tokenized_text'][base_word_idx]))

            getattr(self.model, type(self).model_str).embeddings.word_embeddings.overwrite_fct \
                = lambda embeddings: self._overwrite_embeddings(embeddings, base_word_idx, self.embeddings[base_word])

            logger.debug(
                'Inferring embedding for {} with replacement, base_word_idx = {}'.format(model_input['tokenized_text'],
                                                                                         base_word_idx))

        if len(model_input['masked_indices']) != 1:
            raise ValueError(
                'The pattern must contain exactly one "{}", got "{}" with base word "{}"'.format(
                    self.tokenizer.mask_token, pattern, base_word)
            )

        with torch.no_grad():
            predictions = self.model(
                input_ids=model_input['tokens'].to(self.device),
                token_type_ids=model_input['segments'].to(self.device)
            )[0]

        getattr(self.model, type(self).model_str).embeddings.word_embeddings.overwrite_fct = None
        predicted_tokens = []

        for masked_index in model_input['masked_indices']:
            _, predicted_indices = torch.topk(predictions[0, masked_index], num_predictions)

            for i in range(len(predicted_indices)):
                predicted_index = predicted_indices[i]
                predicted_token = self.tokenizer.convert_ids_to_tokens([predicted_index.item()])[0]
                predicted_tokens.append(predicted_token)
        return predicted_tokens

    def _prepare_text(self, text, use_sep=True, use_cls=True, use_full_stop=True):
        if use_cls:
            text = self.tokenizer.cls_token + ' ' + text
        if use_full_stop and not text[len(text) - 1] in ['?', '.', '!']:
            text += '.'
        if use_sep:
            text += ' ' + self.tokenizer.sep_token

        if isinstance(self.tokenizer, GPT2Tokenizer):
            tokenized_text = self.tokenizer.tokenize(text, add_prefix_space=True)
        else:
            tokenized_text = self.tokenizer.tokenize(text)

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [0] * len(indexed_tokens)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        # get all masked indices
        masked_indices = [i for i, x in enumerate(tokenized_text) if x == self.tokenizer.mask_token]

        return {'tokenized_text': tokenized_text,
                'tokens': tokens_tensor,
                'segments': segments_tensors,
                'masked_indices': masked_indices}

    @staticmethod
    def _overwrite_embeddings(embeddings: torch.Tensor, index: int, replacement_embedding: torch.Tensor):
        # this function is currently not designed to work with more than one batch
        if embeddings.shape[0] != 1:
            raise ValueError('expected a batch of size 1 but found ' + str(embeddings.shape[0].item()))

        embeddings[0, index, :] = replacement_embedding
        return embeddings



class RobertaMaskedLanguageModel(BertMaskedLanguageModel):
    tokenizer_cls = RobertaTokenizer
    model_cls = RobertaForMaskedLM
    model_str = 'roberta'

    def get_predictions(self, pattern: str, base_word: str, num_predictions: int) -> List[str]:
        predictions = super().get_predictions(pattern, base_word, num_predictions)
        return [w.replace('Ġ', '').lower() for w in predictions]
        
        
class DebertaMaskedLanguageModel(BertMaskedLanguageModel):
    tokenizer_cls = DebertaTokenizer
    model_cls = DebertaForMaskedLM
    model_str = 'deberta'

    def get_predictions(self, pattern: str, base_word: str, num_predictions: int) -> List[str]:
        predictions = super().get_predictions(pattern, base_word, num_predictions)
        return [w.replace('Ġ', '').lower() for w in predictions]
        
        
        
#if (args.model_cls == 'bertram'):
    #model = model_cls(args.model_name, 'bert-base-uncased', embeddings)
        
        
class BertramMaskedLanguageModel(CustomMaskedLanguageModel):
    tokenizer_cls = AutoTokenizer
    model_cls = AutoModelForMaskedLM
    model_str = 'bert' #bertram or bertram?
    model_path = './dataset/outputs/BERT_fused_trained-e1/'
    config = AutoConfig.from_pretrained(model_path)

    def get_predictions(self, pattern: str, base_word: str, num_predictions: int) -> List[str]:
        predictions = super().get_predictions(pattern, base_word, num_predictions)
        return [w.replace('Ġ', '').lower() for w in predictions]

class DeBertramaMaskedLanguageModel(CustomMaskedLanguageModel):
    tokenizer_cls = AutoTokenizer
    model_cls = AutoModelForMaskedLM
    model_str = 'deberta' #deberta or debertrama?
    model_path = './dataset/outputs/DeBERTa_fused_trained-e1/'
    config = AutoConfig.from_pretrained(model_path)

    def get_predictions(self, pattern: str, base_word: str, num_predictions: int) -> List[str]:
        predictions = super().get_predictions(pattern, base_word, num_predictions)
        return [w.replace('Ġ', '').lower() for w in predictions]

class RoBertramaMaskedLanguageModel(CustomMaskedLanguageModel):
    tokenizer_cls = AutoTokenizer
    model_cls = AutoModelForMaskedLM
    model_str = 'roberta' #roberta or robertrama?
    model_path = './dataset/outputs/RoBERTa_fused_trained-e1/'
    config = AutoConfig.from_pretrained(model_path)

    def get_predictions(self, pattern: str, base_word: str, num_predictions: int) -> List[str]:
        predictions = super().get_predictions(pattern, base_word, num_predictions)
        return [w.replace('Ġ', '').lower() for w in predictions]

if __name__ == '__main__':
    mlm_bert = BertMaskedLanguageModel('bert-base-uncased')
    mlm_roberta = RobertaMaskedLanguageModel('roberta-base')
    mlm_deberta = DebertaMaskedLanguageModel('microsoft/deberta-base')
    mlm_bertram = BertramMaskedLanguageModel('bertram', 'bert-base-uncased')
    mlm_robertrama = RoBertramaMaskedLanguageModel('robertrama', 'roberta-base')
    mlm_debertrama = DeBertramaMaskedLanguageModel('debertrama','microsoft/deberta-base')


    predictions_bert = mlm_bert.get_predictions(pattern="a <W> is a [MASK]", base_word="lime", num_predictions=10)
    predictions_roberta = mlm_roberta.get_predictions(pattern="a <W> is a [MASK]", base_word="lime", num_predictions=10)
    predictions_deberta = mlm_deberta.get_predictions(pattern="a <W> is a [MASK]", base_word="lime", num_predictions=10)
    
    predictions_bertram = mlm_bertram.get_predictions(pattern="a <W> is a [MASK]", base_word="lime", num_predictions=10)
    predictions_robertrama = mlm_robertrama.get_predictions(pattern="a <W> is a [MASK]", base_word="lime", num_predictions=10)
    predictions_debertrama = mlm_debertrama.get_predictions(pattern="a <W> is a [MASK]", base_word="lime", num_predictions=10)    
    
    
    
    print(predictions_bert)
    print(predictions_roberta)
    print(predictions_deberta)
    print(predictions_bertram)
    print(predictions_robertrama)
    print(predictions_debertrama)
