"""Export the existing analyzer's measured token flow as offline HTML + JSON."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import playground


def export(result):
    data = playground.token_flow_data(result)
    assert len(data['stages']) == 26
    assert all(stage['width'] == 1024 for stage in data['stages'])
    assert all(len(stage['first16']) == len(data['tokens']) for stage in data['stages'])
    assert all(len(vector) == 16 for stage in data['stages'] for vector in stage['first16'])
    target = ROOT / 'docs/examples'
    target.mkdir(exist_ok=True, parents=True)
    (target / 'token-representation.json').write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    (target / 'token-representation.html').write_text(playground.token_flow_html(result))
    return target / 'token-representation.html'


if __name__ == '__main__':
    lab = playground.Playground()
    try:
        print(export(lab.analyze('The capital of Hungary is')))
    finally:
        lab.unload()
