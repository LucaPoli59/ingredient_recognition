import json, sys
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
