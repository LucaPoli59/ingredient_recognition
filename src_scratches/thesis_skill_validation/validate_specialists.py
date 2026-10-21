"""Exercise pinned thesis specialists on public metadata and synthetic artifacts."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skills-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    citation = args.skills_root.resolve() / 'citation-management' / 'scripts'
    visualization = args.skills_root.resolve() / 'scientific-visualization' / 'scripts'
    checks = []

    def run(script, *arguments, expected=0):
        result = subprocess.run(
            [sys.executable, str(script), *map(str, arguments)],
            text=True, encoding='utf-8', capture_output=True, timeout=90,
        )
        assert result.returncode == expected, (script.name, result.stdout, result.stderr)
        checks.append({'command': script.name, 'returncode': result.returncode})
        return result

    bibliography = output / 'references.bib'
    run(citation / 'doi_to_bibtex.py', '10.1109/CVPR.2016.90', '--output', bibliography)
    bibtext = bibliography.read_text(encoding='utf-8')
    assert 'Deep Residual Learning for Image Recognition' in bibtext
    key = re.search(r'@\w+\{([^,]+),', bibtext).group(1)
    before = hashlib.sha256(bibliography.read_bytes()).hexdigest()
    run(citation / 'format_bibtex.py', bibliography, '--output', output / 'formatted.bib')
    assert hashlib.sha256(bibliography.read_bytes()).hexdigest() == before
    assert key in (output / 'formatted.bib').read_text(encoding='utf-8')

    bad = output / 'draft_bad.md'
    bad.write_text(
        '# Synthetic review fixture — not thesis findings\n\n'
        f'Residual learning is introduced by He et al. [@{key}].\n\n'
        'The prototype achieved test macro AP of 0.470, demonstrating robust '
        'generalization across seeds [@MissingStudy2026].\n', encoding='utf-8',
    )
    evidence = {
        'synthetic': True, 'split': 'validation', 'metric': 'label_macro_ap',
        'value': 0.410, 'seeds': [42], 'uncertainty': None,
        'comparators': [], 'purpose': 'Skill review fixture only; no project measurement.',
    }
    (output / 'result.json').write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
    run(citation / 'validate_citations.py', bibliography, '--manuscript', bad,
        '--report', output / 'citation_bad.json', expected=1)
    report = json.loads((output / 'citation_bad.json').read_text(encoding='utf-8'))
    assert report['manuscript_results']['missing_keys'] == ['MissingStudy2026']

    good = output / 'draft_corrected.md'
    good.write_text(
        '# Synthetic review fixture — not thesis findings\n\n'
        f'Residual learning is introduced by He et al. [@{key}].\n\n'
        'In this synthetic fixture, the prototype has validation macro AP of '
        '0.410 for one seed. No evidence of test performance or variation '
        'across training seeds is supplied.\n', encoding='utf-8',
    )
    run(citation / 'validate_citations.py', bibliography, '--manuscript', good,
        '--report', output / 'citation_corrected.json')

    # These are observed limitations, not expected validation successes.
    sys.path.insert(0, str(citation))
    from validate_citations import CitationValidator
    from _common import parse_bibtex
    validator = CitationValidator()
    edge = output / 'citation_edge_cases.md'
    edge.write_text('[-@SuppressedMissing]\n[@Key.WithDot]\n`@CodeExample`\n', encoding='utf-8')
    observed_keys = sorted(validator.parse_manuscript_citations(str(edge)))
    malformed_count = len(parse_bibtex(bibtext + '\n@article{broken, title={unterminated'))

    figure_script = output / 'render_fixture.py'
    figure_script.write_text('''import json, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, sys.argv[1])
from figure_export import export_figure
from style_presets import style_context
out = Path(sys.argv[2])
evidence = json.loads((out / 'result.json').read_text())
with style_context('default', palette_name='okabe_ito_on_white'):
    fig, ax = plt.subplots(figsize=(110/25.4, 65/25.4), layout='constrained')
    ax.bar(['Prototype'], [evidence['value']], color='#0072B2', width=0.45)
    ax.set(ylim=(0, 1), ylabel='Validation macro AP', title='Synthetic review fixture')
    ax.text(0, evidence['value'] + 0.025, f"{evidence['value']:.3f}", ha='center')
    export_figure(fig, out / 'fixture', formats=['png', 'pdf'], dpi=300,
                  bbox_inches=None, write_manifest=True,
                  provenance={'raw_data': 'result.json', 'transformations': [],
                              'uncertainty': 'not supplied; one synthetic seed'})
    plt.close(fig)
''', encoding='utf-8')
    run(figure_script, visualization, output)
    run(visualization / 'image_metadata.py', output / 'fixture.png',
        '--format', 'png', '--target-width-mm', '110', '--output', output / 'png_metadata.json')
    run(visualization / 'image_metadata.py', output / 'fixture.pdf',
        '--format', 'pdf', '--output', output / 'pdf_metadata.json')
    palette = run(visualization / 'palette_audit.py', '--palette', 'okabe_ito_on_white',
                  '--background', 'FFFFFF', '--role', 'graphical')
    (output / 'palette_audit.json').write_text(palette.stdout, encoding='utf-8')
    summary = {
        'python': sys.version.split()[0],
        'packages': {name: importlib.metadata.version(name) for name in
                     ['requests', 'matplotlib', 'pillow', 'pypdf', 'pyyaml']},
        'checks': checks,
        'bibliography_input_unchanged': True,
        'citation_key_preserved': key,
        'observed_edge_keys': observed_keys,
        'entries_after_appending_unterminated_entry': malformed_count,
        'numeric_and_scope_review': 'Manual review required; see review.md. Citation scripts do not check numbers.',
    }
    (output / 'validation.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
