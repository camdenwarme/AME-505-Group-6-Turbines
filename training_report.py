"""Write a standalone HTML report with the energy and loss plots stacked."""

import argparse
import base64
from html import escape
from pathlib import Path


def write_training_report(run_dir):
    run_dir = Path(run_dir)
    plots = []
    for filename, title in (
        ("cumulative_energy_by_episode.png", "Energy generation by episode"),
        ("training_loss_by_episode.png", "Training loss by episode"),
    ):
        # Embed the original PNG bytes: the HTML remains usable when copied
        # independently of the run folder, without needing a server or network.
        encoded = base64.b64encode((run_dir / filename).read_bytes()).decode("ascii")
        plots.append(f'<section><h2>{escape(title)}</h2>'
                     f'<img src="data:image/png;base64,{encoded}" '
                     f'alt="{escape(title)}"></section>')
    report = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Wind-farm training plots</title>
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; background: #eef2f6; color: #172536;
           font-family: system-ui, sans-serif; }
    main { max-width: 1664px; margin: 0 auto; padding: 28px 24px; }
    h1 { margin: 0 0 8px; font-size: 28px; }
    .run { margin: 0 0 28px; color: #536276; overflow-wrap: anywhere; }
    section { margin: 0 0 28px; padding: 16px; background: white;
              border: 1px solid #d9e1ea; border-radius: 10px; }
    h2 { margin: 0 0 12px; font-size: 19px; }
    img { display: block; width: 100%; height: auto; }
    @media (max-width: 600px) {
      main { padding: 18px 8px; }
      section { padding: 8px; }
    }
  </style>
</head>
<body><main>
  <h1>Wind-farm training plots</h1>
  <p class="run">RUN_NAME</p>
  PLOTS
</main></body>
</html>
"""
    report = report.replace("RUN_NAME", escape(f"{run_dir.parent.name} / {run_dir.name}"))
    report = report.replace("PLOTS", "\n".join(plots))
    output = run_dir / "training_plots.html"
    output.write_text(report, encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_folder", type=Path)
    print(write_training_report(parser.parse_args().run_folder))
