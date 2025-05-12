# stdlib
from dataclasses import dataclass
import errno
import os
import logging

# third-party
from dataclasses_json import dataclass_json
import numpy as np
import matplotlib.pyplot as plt

# first-party
from .alt_types import SongInfo, VadResult

logger = logging.getLogger(__name__)


@dataclass_json
@dataclass
class HtmlAlignmentChunk:
    chunk_type: str  # E/S/I/D
    ref_text: str  # ++ for insertion
    hyp_text: str  # -- for deletion
    timestamp: float


@dataclass_json
@dataclass
class HtmlSongSummary:
    song_id: str
    uid: str
    language: str
    wer: float
    wil: float
    wer_near: float
    hits: int
    subs: int
    ins: int
    dels: int
    nears: int
    num_ref_tokens: int


@dataclass_json
@dataclass
class ErrorRates:
    wer: float
    wer_total: float = 0
    wil: float | None = None
    wer_near: float | None = None


@dataclass_json
@dataclass
class HtmlSummary:
    title: str
    description: str
    mean_wer: float
    mean_wil: float
    mean_wer_near: float
    lang_error_rates: dict[str, ErrorRates]
    lang_ds_error_rates: dict[str, dict[str, ErrorRates]]
    ds_error_rates: dict[str, ErrorRates]
    song_summaries: list[HtmlSongSummary]
    wer_total: float = 0


# write summary table out to a html file
def render_summary_html(output_fn: str, summary: HtmlSummary):
    summary.song_summaries.sort(key=lambda res: -res.wer)  # highest wer first
    html_header = """
    <head>
    <title>Transcription Summary</title>
    <style>
    body {
        font-family: courier
    }
    .divider {
        padding: 0;
    }
    .divider hr {
        margin: 0;
        border: none;
        border-top: 1px solid #ddd;
    }
    </style>
    </head>
    """
    html_body = [
        f"""
    <body>
    <h1>Summary: {summary.title}</h1>
    <h3>"Pipeline information: {summary.description}"</h1>
    <h2>Headline WER: {summary.mean_wer}</h2>
    <table>
    <tr>
      <th>Song</th>
      <th>wer</th>
      <th>wer_total</th>
    </tr>
    """
    ]
    table_divider = "<tr><td class='divider' colspan=4><hr/></td></tr>"
    html_body.append("<tr>")
    html_body.append("<td><b>Means</b></td>")
    html_body.append(f"<td>{summary.mean_wer:.2f}</td>")
    html_body.append(f"<td>{summary.wer_total:.2f}</td>")
    html_body.append("</tr>")
    html_body.append(table_divider)
    if len(summary.lang_error_rates) > 1:
        for lang, rates in summary.lang_error_rates.items():
            html_body.append("<tr>")
            html_body.append(f"<td><b>{lang}</b></td>")
            html_body.append(f"<td>{rates.wer:.2f}</td>")
            html_body.append(f"<td>{rates.wer_total:.2f}</td>")
            html_body.append("</tr>")
        html_body.append(table_divider)
    if len(summary.ds_error_rates) > 1:
        for ds, rates in summary.ds_error_rates.items():
            html_body.append("<tr>")
            html_body.append(f"<td><b>{ds}</b></td>")
            html_body.append(f"<td>{rates.wer:.2f}</td>")
            html_body.append(f"<td>{rates.wer_total:.2f}</td>")
            html_body.append("</tr>")
        html_body.append(table_divider)
    for song in summary.song_summaries:
        html_body.append("<tr>")
        html_body.append(f"<td><a href={song.uid}.html>{song.uid}</a></td>")
        html_body.append(f"<td>{song.wer:.2f}</td>")
        html_body.append("</tr>")
    html_body.append("</table>")
    html_body.append("</body>")

    # write list of tags
    with open(output_fn, "w") as fout:
        fout.write("<!DOCTYPE html>")
        fout.write(html_header)
        for tag in html_body:
            fout.write(tag)


def vad_plot(fn: str, vad_result: VadResult):
    plt.figure()
    active = np.zeros(len(vad_result.scores))
    for segment in vad_result.segments:
        start_frame = segment["start"] // vad_result.window_size_samples
        end_frame = segment["end"] // vad_result.window_size_samples
        active[start_frame:end_frame] = 1
    # 512 is vad window_size_samples
    plt.plot(vad_result.scores, label="RMS")
    plt.plot(active, label="Clips")
    plt.legend()
    plt.savefig(fn)
    plt.close()


def symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e


def render_song_html(
    songinfo: SongInfo,
    html_verses: list[list[list[HtmlAlignmentChunk]]],
    eval_dir: str,
):
    """
    Generates an HTML file that visualizes the transcription results with audio playback functionality.

    Args:
        result (jiwer.WordOutput): The output from jiwer.process_words
        songinfo (Dict): song info json from evaluate.py
        html_fn (str): The file name (including path) where the generated HTML will be saved.
        audio_fn (str): The file name (including path) of the audio file to be used for
            playback within the HTML.

    Returns:
        None: This function does not return any value. It creates an HTML file with the
        specified file name.
    """

    html_fn = os.path.join(eval_dir, songinfo.extract.uid + ".html")
    html_audio_fn = os.path.join(eval_dir, songinfo.extract.uid + ".mp3")
    # symlink_force(songinfo.extract.audio_fn, html_audio_fn)
    assert songinfo.infer is not None
    if songinfo.infer.infer_target != "original":
        html_vocals_fn = os.path.join(
            eval_dir, f"{songinfo.extract.uid}_{songinfo.infer.infer_target}.mp3"
        )
        # vocals_fn = songinfo.infer.audio_fn
        # if str(songinfo.infer.audio_fn).startswith("build/"):
        #     vocals_fn = os.path.join(os.getcwd(), songinfo.infer.audio_fn)
        # symlink_force(vocals_fn, html_vocals_fn)
    else:
        html_vocals_fn = None
    html_header = f"""
    <head>
    <title>Transcription: {html_audio_fn}</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/wavesurfer.js/7.8.7/wavesurfer.min.js"></script>
    <style>
    body {{
        font-family: courier
    }}
    .waveform {{
        margin: 20px 0;
        padding: 20px;
        border: 1px solid #ddd;
        border-radius: 4px;
    }}
    .alignment {{
        font-family: courier;
        margin:-2px
    }}
    .line {{
        clear: both;
        margin: 5px;
    }}
    .chunk {{
        float: left;
        margin: 2px;
    }}
    .hyp.delete a, .hyp.insert a, .hyp.substitute a {{
        color: red;
        text-decoration: underline;
    }}
    .hyp:hover {{
        color: white;
        background-color: red;
    }}
    </style>
    </head>
    """
    html_tags = ["<!DOCTYPE html>", "<html>", html_header]
    html_tags.append("<body>")
    html_tags.append(f"<h1>{os.path.basename(html_audio_fn)}</h1>")
    # Links to other files
    html_tags.append('<h3><a href="summary.html">Summary backlink</a></h2>')
    html_tags.append(f'<h3><a href="{songinfo.extract.uid}.json">JSON summary</a></h2>')
    raw_lyrics_fn = os.path.join(eval_dir, songinfo.extract.uid + ".lyrics.txt")
    with open(raw_lyrics_fn, "w") as fout:
        fout.write(songinfo.extract.lyrics.text)
    html_tags.append(
        f'<h3><a href="{os.path.basename(raw_lyrics_fn)}">raw_lyrics</a></h2>'
    )

    # Audio players
    # Two waveforms with common controls to play/pause/toggle separation
    html_tags.append('<div class="waveform"><div id="ws-mixed"></div></div>')
    hidden = "hidden" if songinfo.infer.infer_target == "original" else ""
    html_tags.append(f'<div {hidden} class="waveform"><div id="ws-vocals"></div></div>')
    html_tags.append(
        f"""
    <div {hidden} class="controls">
        <button onclick="togglePlay()">Play/Pause</button>
        <button onclick="toggleVocals()">Vocals/Mixed</button>
    </div>
    """
    )
    # Script to specify waveform controller behavior
    common_ws_options = """
        width: 800,
        height: 100,
        normalize: true,
        mediaControls: true,
    """
    active_ws_colors = """
        waveColor: '#4F4A85',
        progressColor: '#383351',
    """
    inactive_ws_colors = """
        waveColor: '#808080',
        progressColor: '#606060',
    """
    html_tags.append(
        f"""
    <script>
        var vocals = false;
        const ws_mixed = WaveSurfer.create({{
            container:"#ws-mixed",
            url:'/{html_audio_fn}',
            {common_ws_options}
            {active_ws_colors}
        }}) 
        const ws_vocals = WaveSurfer.create({{
            container:"#ws-vocals",
            url:'/{html_vocals_fn}',
            {common_ws_options}
            {inactive_ws_colors}
        }}) 
        // play/pause toggle function
        function togglePlay() {{
            if (ws_vocals.isPlaying()) {{
                ws_vocals.pause();
                ws_mixed.pause();
            }}
            else {{
                ws_vocals.play();
                ws_mixed.play();
            }}
        }}
        function toggleVocals() {{
            if (vocals) {{
                ws_vocals.setVolume(0);
                ws_vocals.setOptions({{
                    {inactive_ws_colors}
                }})
                ws_mixed.setVolume(1);
                ws_mixed.setOptions({{
                    {active_ws_colors}
                }})
            }}
            else {{
                ws_mixed.setVolume(0);
                ws_mixed.setOptions({{
                    {inactive_ws_colors}
                }})
                ws_vocals.setVolume(1);
                ws_vocals.setOptions({{
                    {active_ws_colors}
                }})
            }}
            vocals = ! vocals;
        }}
    </script> """
    )
    assert songinfo.infer is not None
    if songinfo.infer.vad_result is not None:
        vad_plot_fn = songinfo.extract.song_id + "_vad.svg"
        vad_plot(
            os.path.join(eval_dir, vad_plot_fn), vad_result=songinfo.infer.vad_result
        )
        html_tags.append(f'<div><img src="{vad_plot_fn}" width=800></img></div>')

    html_tags.append("<div class='alignment'>")
    for html_lines in html_verses:
        html_tags.append("<div class='verse'>")
        for line in html_lines:
            html_tags.append("<div class='line'>")
            for chunk in line:
                onclick = f"""
                ws_mixed.seekTo({chunk.timestamp}/ws_mixed.getDuration());
                ws_mixed.play();
                """
                if html_vocals_fn is not None:
                    onclick += f"""
                    ws_vocals.seekTo({chunk.timestamp}/ws_mixed.getDuration());
                    ws_vocals.play();
                    """
                chunk_html = f"""
                <div class="chunk {chunk.chunk_type}">
                <span class="ref {chunk.chunk_type}">{chunk.ref_text}</span><br/>
                <span class="hyp {chunk.chunk_type}">
                    <a href="javascript:;" onclick="{onclick}">{chunk.hyp_text}</a>
                </span>
                </div>"""
                html_tags.append(chunk_html)
            html_tags.append("</div>")  # line
        html_tags.append("</div>")  # verse
        html_tags.append("<hr>")  # section divider between verses
    html_tags.append("</div>")  # alignment
    html_tags.append("</body>")
    html_tags.append("</html>")
    with open(html_fn, "w") as fout:
        fout.write("".join(html_tags))
