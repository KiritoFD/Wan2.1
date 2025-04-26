# 此文件的内容应与原 tokenizers.py 完全相同
# 只需将文件重命名即可
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import html
import string

import ftfy
import regex as re
# 移除顶层导入，改为在类中动态导入
# from transformers import AutoTokenizer

__all__ = ['HuggingfaceTokenizer']


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def canonicalize(text, keep_punctuation_exact_string=None):
    text = text.replace('_', ' ')
    if keep_punctuation_exact_string:
        text = keep_punctuation_exact_string.join(
            part.translate(str.maketrans('', '', string.punctuation))
            for part in text.split(keep_punctuation_exact_string))
    else:
        text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


class HuggingfaceTokenizer:

    def __init__(self, name, seq_len=None, clean=None, **kwargs):
        assert clean in (None, 'whitespace', 'lower', 'canonicalize')
        self.name = name
        self.seq_len = seq_len
        self.clean = clean

        # 动态导入 AutoTokenizer，避免循环导入
        from transformers import AutoTokenizer
        
        # 检查是否为本地路径（包含目录分隔符或文件扩展名）
        import os
        is_local_path = os.path.exists(name) or ('\\' in name) or ('/' in name)
        
        # init tokenizer - 根据不同情况处理路径
        try:
            if is_local_path:
                # 修复Windows路径分隔符问题
                name = os.path.normpath(name)
                if os.path.exists(name):
                    # 确认路径存在，使用本地路径加载
                    self.tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True, **kwargs)
                else:
                    # 如果路径不存在，记录错误并尝试回退
                    import logging
                    logging.warning(f"Tokenizer路径不存在: {name}，尝试作为repo_id加载")
                    self.tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)
            else:
                # 使用默认行为 (repo_id)
                self.tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)
            
            self.vocab_size = self.tokenizer.vocab_size
            
        except Exception as e:
            # 提供更友好的错误信息
            import logging
            logging.error(f"Tokenizer加载失败: {str(e)}")
            logging.error(f"尝试加载的路径/ID: {name}")
            logging.error(f"是否被视为本地路径: {is_local_path}")
            # 重新抛出异常
            raise

    def __call__(self, sequence, **kwargs):
        return_mask = kwargs.pop('return_mask', False)

        # arguments
        _kwargs = {'return_tensors': 'pt'}
        if self.seq_len is not None:
            _kwargs.update({
                'padding': 'max_length',
                'truncation': True,
                'max_length': self.seq_len
            })
        _kwargs.update(**kwargs)

        # tokenization
        if isinstance(sequence, str):
            sequence = [sequence]
        if self.clean:
            sequence = [self._clean(u) for u in sequence]
        ids = self.tokenizer(sequence, **_kwargs)

        # output
        if return_mask:
            return ids.input_ids, ids.attention_mask
        else:
            return ids.input_ids

    def _clean(self, text):
        if self.clean == 'whitespace':
            text = whitespace_clean(basic_clean(text))
        elif self.clean == 'lower':
            text = whitespace_clean(basic_clean(text)).lower()
        elif self.clean == 'canonicalize':
            text = canonicalize(basic_clean(text))
        return text
