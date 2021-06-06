from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForMaskedLM,
    BertModel,
    BertForSequenceClassification,
    BertTokenizer,
    CamembertConfig,
    CamembertForMaskedLM,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForMaskedLM,
    DistilBertTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTConfig,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaModel,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    ElectraConfig,
    ElectraModel,
    ElectraForSequenceClassification,
    ElectraTokenizer,
    BartConfig,
    # BartForConditionalGeneration,
    BartTokenizer,
)

from .modeling_bart import BartForConditionalGeneration

MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    "openai-gpt": (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "bert": (BertConfig, BertModel, BertTokenizer),
    "roberta": (RobertaConfig, RobertaModel, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    "camembert": (CamembertConfig, CamembertForMaskedLM, CamembertTokenizer),
    "electra": (ElectraConfig, ElectraModel, ElectraTokenizer),
}

MODEL_CLASSES_FOR_CLASSIFICATION = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "electra": (ElectraConfig, ElectraForSequenceClassification, ElectraTokenizer),
}

MODEL_CLASSES_FOR_GENERATION = {
    "bart": (BartConfig, BartForConditionalGeneration, BartTokenizer),
}

def load_pretrain_lm(pretrain_model_name_or_path, **kwargs):
    model_type = None
    if 'bart' in pretrain_model_name_or_path.lower():
        model_type = 'bart'
    if model_type is None:
        raise RuntimeError()
    else:
        config_class, model_class, tokenizer_class = MODEL_CLASSES_FOR_GENERATION[model_type]
        config = config_class.from_pretrained(pretrain_model_name_or_path)
        for key, value in kwargs.items():
            setattr(config, key, value)
        model = model_class.from_pretrained(pretrain_model_name_or_path, config=config)
        return config, model

def load_pretrain_encoder(pretrain_model_name_or_path, use_model_for_classification=False, **kwargs):
    model_type = None
    if 'roberta' in pretrain_model_name_or_path.lower():
        model_type = 'roberta'
    elif 'albert' in pretrain_model_name_or_path.lower():
        model_type = 'albert'
    elif 'bert' in pretrain_model_name_or_path.lower():
        model_type = 'bert'
    elif 'electra' in pretrain_model_name_or_path.lower():
        model_type = 'electra'
    if model_type is None:
        raise RuntimeError()
    else:
        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type] if not use_model_for_classification else MODEL_CLASSES_FOR_CLASSIFICATION[model_type]
        config = config_class.from_pretrained(pretrain_model_name_or_path)
        config.output_hidden_states = True
        config.is_decoder = False
        for key, value in kwargs.items():
            setattr(config, key, value)
        model = model_class.from_pretrained(pretrain_model_name_or_path, config=config)
        return config, model

def set_embeddings(pretrain_model, embeddings):
    if hasattr(pretrain_model, 'embeddings'):
        pretrain_model.embeddings = embeddings
    elif hasattr(pretrain_model, 'bert'):
        pretrain_model.bert.embeddings = embeddings
    elif hasattr(pretrain_model, 'electra'):
        pretrain_model.electra.embeddings = embeddings
    else:
        raise RuntimeError()

def get_embeddings(pretrian_model):
    if hasattr(pretrian_model, 'embeddings'):
        return pretrian_model.embeddings
    elif hasattr(pretrian_model, 'bert'):
        return pretrian_model.bert.embeddings
    elif hasattr(pretrian_model, 'electra'):
        return pretrian_model.electra.embeddings
    else:
        raise RuntimeError()
