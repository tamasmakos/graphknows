from pathlib import Path

from graphgen.prototype_gliner2_runner import run_gliner2_ontology_prototype, load_default_ontology_labels


def main() -> None:
    input_path = Path('/root/graphgen/input/txt/translated/CRE-9-2022-05-03-ITM-005_EN.txt')
    line = input_path.read_text(encoding='utf-8').splitlines()[0]
    labels = load_default_ontology_labels()
    result = run_gliner2_ontology_prototype(
        text=line,
        ontology_labels=labels,
        output_dir='/root/graphgen/output_gliner2_prototype',
        top_k=4,
    )
    print(result)


if __name__ == '__main__':
    main()
