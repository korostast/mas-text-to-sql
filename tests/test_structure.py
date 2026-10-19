"""Protect package boundaries, source fingerprints, and code-style requirements."""
import ast
import importlib
import pkgutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mas_sql
from mas_sql.cli import source_hash


class PackageStructureTests(unittest.TestCase):
    """Keep the decomposed package importable and consistently documented."""

    def test_public_root_has_only_entry_points(self) -> None:
        """Reject reintroduced flat implementation modules."""
        root = Path(mas_sql.__file__).parent
        self.assertEqual({p.name for p in root.glob('*.py')},
                         {'__init__.py', '__main__.py', 'cli.py'})
        for name in ('agent', 'llm', 'database', 'data', 'evaluation'):
            self.assertTrue((root / name / '__init__.py').is_file(), name)

    def test_all_package_modules_import_without_running_commands(self) -> None:
        """Import every module without starting CLI or worker execution."""
        for module in pkgutil.walk_packages(mas_sql.__path__, 'mas_sql.'):
            with self.subTest(module=module.name):
                importlib.import_module(module.name)

    def test_package_functions_have_annotations_and_docstrings(self) -> None:
        """Require docstrings and explicit callable boundary annotations."""
        root = Path(mas_sql.__file__).parent
        for path in sorted(root.rglob('*.py')):
            tree = ast.parse(path.read_text(encoding='utf-8'))
            with self.subTest(module=str(path.relative_to(root))):
                self.assertTrue(ast.get_docstring(tree), 'Missing module docstring')
            for node in ast.walk(tree):
                if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                with self.subTest(module=str(path.relative_to(root)), name=node.name, line=node.lineno):
                    self.assertTrue(ast.get_docstring(node), 'Missing docstring')
                    if isinstance(node, ast.ClassDef):
                        continue
                    self.assertIsNotNone(node.returns, 'Missing return annotation')
                    arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
                    if node.args.vararg:
                        arguments.append(node.args.vararg)
                    if node.args.kwarg:
                        arguments.append(node.args.kwarg)
                    for argument in arguments:
                        if argument.arg not in {'self', 'cls'}:
                            self.assertIsNotNone(argument.annotation, f'Missing annotation: {argument.arg}')

    def test_source_hash_includes_nested_contents_and_relative_paths(self) -> None:
        """Detect nested source edits and moves even with identical filenames."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / 'cli.py'
            entry.write_text('# entry\n', encoding='utf-8')
            first = root / 'agent' / 'types.py'
            first.parent.mkdir()
            first.write_text('# first\n', encoding='utf-8')
            second = root / 'llm' / 'types.py'
            second.parent.mkdir()
            second.write_text('# second\n', encoding='utf-8')
            with patch('mas_sql.cli.__file__', str(entry)):
                initial = source_hash()
                first.write_text('# changed\n', encoding='utf-8')
                changed = source_hash()
                self.assertNotEqual(initial, changed)
                first.rename(first.with_name('models.py'))
                self.assertNotEqual(changed, source_hash())
                stable = source_hash()
                (root / 'notes.txt').write_text('not Python source', encoding='utf-8')
                self.assertEqual(stable, source_hash())
