import pytest
from datetime import date
from src.kg.graph.parsers.multilingual import MultilingualParser

def test_multilingual_parser_english():
    parser = MultilingualParser()
    content = "seg1\tHello world, this is English."
    filename = "test_2024-01-01.txt"
    doc_date = date(2024, 1, 1)
    
    segments = parser.parse(content, filename, doc_date)
    assert len(segments) == 1
    assert segments[0].metadata['language'] == 'en'

def test_multilingual_parser_chinese():
    parser = MultilingualParser()
    content = "seg1\t你好世界，这是中文测试。"
    filename = "test_2024-01-01.txt"
    doc_date = date(2024, 1, 1)
    
    segments = parser.parse(content, filename, doc_date)
    assert len(segments) == 1
    assert segments[0].metadata['language'] == 'zh'

def test_multilingual_parser_mixed():
    parser = MultilingualParser()
    content = "seg1\tHello\nseg2\t你好"
    filename = "test_2024-01-01.txt"
    doc_date = date(2024, 1, 1)
    
    segments = parser.parse(content, filename, doc_date)
    assert len(segments) == 2
    assert segments[0].metadata['language'] == 'en'
    assert segments[1].metadata['language'] == 'zh'
