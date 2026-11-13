"""Write standalone HTML reports for training plots and evaluation states."""

import argparse
import base64
from html import escape
from pathlib import Path
import re


def _plot_image(path, title):
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return (f'<img src="data:image/png;base64,{encoded}" '
            f'alt="{escape(title)}">')


def write_training_report(run_dir):
    run_dir = Path(run_dir)
    plots = []
    for filename, title in (
        ("cumulative_energy_by_episode.png", "Energy generation by episode"),
        ("training_loss_by_episode.png", "Training loss by episode"),
    ):
        # Embed the original PNG bytes: the HTML remains usable when copied
        # independently of the run folder, without needing a server or network.
        plots.append(f'<section><h2>{escape(title)}</h2>'
                     f'{_plot_image(run_dir / filename, title)}</section>')
    if (run_dir / "reward_by_episode.png").exists():
        plots.append('<section><h2>Reward by episode</h2>' +
                     _plot_image(run_dir / "reward_by_episode.png", "Reward by episode") + '</section>')
    return _write_report(run_dir, "Wind-farm training plots", "training_plots.html", plots)


def write_evaluation_state_report(run_dir):
    run_dir = Path(run_dir)
    episodes = {}
    for path in run_dir.glob("turbine*_states_episode*.png"):
        match = re.fullmatch(r"turbine(\d+)_states_episode(\d+)\.png", path.name)
        if match:
            agent, episode = map(int, match.groups())
            episodes.setdefault(episode, []).append((agent, path))
    if not episodes:
        raise ValueError("No evaluation state-history plots were found in the run folder.")
    sections = ['<nav aria-label="Evaluation episodes">Episodes: ' + " · ".join(
        f'<a href="#episode{episode}">{episode}</a>' for episode in sorted(episodes)) + '</nav>']
    if (run_dir / "reward_by_episode.png").exists():
        sections.append('<section><h2>Reward by episode</h2>' +
                        _plot_image(run_dir / "reward_by_episode.png", "Reward by episode") + '</section>')
    for episode, agents in sorted(episodes.items()):
        plots = [f'<section id="episode{episode}"><h2>Evaluation episode {episode}</h2>']
        for agent, path in sorted(agents):
            title = f"Turbine {agent} states — evaluation episode {episode}"
            plots.append(f'<h3>Turbine {agent}</h3>{_plot_image(path, title)}')
        reward_plot = run_dir / f"reward_history_episode{episode}.png"
        energy_plot = run_dir / f"farm_energy_history_episode{episode}.png"
        if reward_plot.exists() or energy_plot.exists():
            plots.append('<h3>Farm history</h3>')
        if reward_plot.exists():
            plots.append('<h4>Reward</h4>' +
                         _plot_image(reward_plot, f"Reward history — evaluation episode {episode}"))
        if energy_plot.exists():
            plots.append('<h4>Cumulative energy generation</h4>' +
                         _plot_image(energy_plot, f"Farm energy history — evaluation episode {episode}"))
        plots.append('</section>')
        sections.append("\n".join(plots))
    return _write_report(run_dir, "Evaluation states and farm history (greedy policy, no exploration)",
                         "evaluation_states.html", sections)


def _write_report(run_dir, title, filename, sections):
    report = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>REPORT_TITLE</title>
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
    h3 { margin: 20px 0 8px; font-size: 16px; }
    h4 { margin: 16px 0 8px; font-size: 15px; }
    nav { margin-bottom: 24px; line-height: 2; }
    a { color: #1768ac; }
    img { display: block; width: 100%; height: auto; }
    @media (max-width: 600px) {
      main { padding: 18px 8px; }
      section { padding: 8px; }
    }
  </style>
</head>
<body><main>
  <h1>REPORT_TITLE</h1>
  <p class="run">RUN_NAME</p>
  PLOTS
</main></body>
</html>
"""
    report = report.replace("PLOTS", "\n".join(sections))
    report = report.replace("REPORT_TITLE", escape(title))
    report = report.replace("RUN_NAME", escape(f"{run_dir.parent.name} / {run_dir.name}"))
    output = run_dir / filename
    output.write_text(report, encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_folder", type=Path)
    parser.add_argument("--evaluation-states", action="store_true",
                        help="Build the evaluation-state report instead of the training-plot report.")
    args = parser.parse_args()
    writer = write_evaluation_state_report if args.evaluation_states else write_training_report
    print(writer(args.run_folder))
