import unittest
import os
from src.kg.config import KGConfig, get_shared_config, should_process_file

class TestKGConfig(unittest.TestCase):
    def test_default_config(self):
        config = KGConfig()
        self.assertTrue(config.get('input_dir').endswith("input/txt/translated"))
        self.assertTrue(config.get('output_dir').endswith("output"))
        self.assertEqual(config.get('speech_limit'), 10)

    def test_config_overrides(self):
        overrides = {"input_dir": "test_input", "speech_limit": 10}
        config = KGConfig(**overrides)
        self.assertEqual(config.get('input_dir'), "test_input")
        self.assertEqual(config.get('speech_limit'), 10)

    def test_get_shared_config(self):
        config = get_shared_config(input_dir="shared_input")
        self.assertEqual(config.get('input_dir'), "shared_input")
        self.assertTrue(config.get('output_dir').endswith("output"))

    def test_should_process_file(self):
        config = KGConfig()
        # Mock regex to match test.txt
        config.set('file_regex', r'test\.txt')
        self.assertTrue(should_process_file("test.txt", config))
        self.assertFalse(should_process_file("other.txt", config))

if __name__ == '__main__':
    unittest.main()
