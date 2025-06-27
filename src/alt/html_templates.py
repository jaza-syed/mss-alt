SONG_HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<title>Transcription: {{ html_audio_fn }}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/wavesurfer.js/7.8.7/wavesurfer.min.js"></script>
<style>
body {
    font-family: 'Courier New', monospace;
    background-color: #f8f9fa;
    margin: 0;
    padding: 15px;
    line-height: 1.4;
}

h1 {
    color: #2c3e50;
    text-align: center;
    margin: 0 0 8px 0;
    font-size: 1.8em;
}

h3 {
    margin: 6px 0;
}

h3 a {
    color: #3498db;
    text-decoration: none;
    padding: 8px 16px;
    background-color: #ecf0f1;
    border-radius: 4px;
    display: inline-block;
    transition: all 0.3s ease;
}

h3 a:hover {
    background-color: #3498db;
    color: white;
}

.audio-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    margin: 12px 0;
}

.waveform {
    width: 100%;
    max-width: 800px;
    margin: 6px 0;
    padding: 15px;
    border: 2px solid #bdc3c7;
    border-radius: 8px;
    background-color: white;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.controls {
    text-align: center;
    margin: 12px 0;
}

.controls button {
    background-color: #3498db;
    color: white;
    border: none;
    padding: 8px 16px;
    margin: 0 8px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 14px;
    transition: background-color 0.3s ease;
}

.controls button:hover {
    background-color: #2980b9;
}

.alignment {
    font-family: 'Courier New', monospace;
    margin: 15px 0;
    background-color: white;
    border-radius: 8px;
    padding: 15px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.sample {
    margin-bottom: 15px;
    padding: 10px;
    background-color: #fdfdfd;
    border: 1px solid #e1e8ed;
    border-radius: 6px;
}

.line {
    display: flex;
    flex-wrap: wrap;
    margin: 4px 0;
    align-items: flex-start;
}

.chunk {
    margin: 2px 6px 2px 0;
    padding: 6px;
    border-radius: 4px;
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    min-width: 80px;
    transition: all 0.2s ease;
}

.chunk:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
}

.chunk.delete {
    background-color: #fee;
    border-color: #fcc;
}

.chunk.insert {
    background-color: #efe;
    border-color: #cfc;
}

.chunk.substitute {
    background-color: #fff3cd;
    border-color: #ffeaa7;
}

.ref {
    display: block;
    font-weight: bold;
    color: #2c3e50;
    margin-bottom: 1px;
    font-size: 0.85em;
    line-height: 1.1;
}

.hyp {
    display: block;
    font-size: 0.85em;
    line-height: 1.1;
    margin-top: 1px;
}

.hyp a {
    color: #34495e;
    text-decoration: none;
    transition: all 0.2s ease;
}

.hyp.delete a, .hyp.insert a, .hyp.substitute a {
    color: #e74c3c;
    font-weight: bold;
}

.hyp a:hover {
    color: white;
    background-color: #3498db;
    padding: 2px 4px;
    border-radius: 3px;
}

.hyp.delete a:hover, .hyp.insert a:hover, .hyp.substitute a:hover {
    background-color: #e74c3c;
}

hr {
    border: none;
    height: 2px;
    background: linear-gradient(to right, #3498db, #e74c3c, #3498db);
    margin: 10px 0;
    border-radius: 1px;
}

.vad-plot {
    text-align: center;
    margin: 20px 0;
}

.vad-plot img {
    max-width: 100%;
    max-width: 800px;
    height: auto;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

/* Responsive design */
@media (max-width: 768px) {
    body {
        padding: 10px;
    }
    
    .waveform {
        padding: 10px;
    }
    
    .line {
        flex-direction: column;
    }
    
    .chunk {
        margin-bottom: 8px;
        width: 100%;
    }
    
    .controls button {
        display: block;
        width: 100%;
        margin: 5px 0;
    }
}
</style>
</head>
<body>
<h1>{{ html_audio_fn }}</h1>
<h3><a href="summary.html">Summary backlink</a></h3>
<h3><a href="{{ song.uid }}.json">JSON summary</a></h3>
<h3><a href="{{ raw_lyrics_fn }}">raw_lyrics</a></h3>

<div class="audio-container">
    <div class="waveform"><div id="ws-mixed"></div></div>
    {% if html_vocals_fn %}
    <div class="waveform"><div id="ws-vocals"></div></div>
    <div class="controls">
        <button onclick="togglePlay()">Play/Pause</button>
        <button onclick="toggleVocals()">Vocals/Mixed</button>
    </div>
    {% endif %}
</div>

<script>
    var vocals = false;
    const ws_mixed = WaveSurfer.create({
        container:"#ws-mixed",
        url:'/{{ html_audio_fn }}',
        width: 800,
        height: 100,
        normalize: true,
        mediaControls: true,
        waveColor: '#4F4A85',
        progressColor: '#383351',
    }) 
    {% if html_vocals_fn %}
    const ws_vocals = WaveSurfer.create({
        container:"#ws-vocals",
        url:'/{{ html_vocals_fn }}',
        width: 800,
        height: 100,
        normalize: true,
        mediaControls: true,
        waveColor: '#808080',
        progressColor: '#606060',
    }) 
    function togglePlay() {
        if (ws_vocals.isPlaying()) {
            ws_vocals.pause();
            ws_mixed.pause();
        }
        else {
            ws_vocals.play();
            ws_mixed.play();
        }
    }
    function toggleVocals() {
        if (vocals) {
            ws_vocals.setVolume(0);
            ws_vocals.setOptions({
                waveColor: '#808080',
                progressColor: '#606060',
            })
            ws_mixed.setVolume(1);
            ws_mixed.setOptions({
                waveColor: '#4F4A85',
                progressColor: '#383351',
            })
        }
        else {
            ws_mixed.setVolume(0);
            ws_mixed.setOptions({
                waveColor: '#808080',
                progressColor: '#606060',
            })
            ws_vocals.setVolume(1);
            ws_vocals.setOptions({
                waveColor: '#4F4A85',
                progressColor: '#383351',
            })
        }
        vocals = ! vocals;
    }
    {% endif %}
</script>

{% if vad_plot_fn %}
<div class="vad-plot">
    <img src="{{ vad_plot_fn }}" alt="VAD Plot">
</div>
{% endif %}

<div class='alignment'>
{% for html_sample in html_samples %}
    <div class='sample'>
    {% for line in html_sample %}
        <div class='line'>
        {% for chunk in line %}
            <div class="chunk {{ chunk.chunk_type }}">
                <span class="ref {{ chunk.chunk_type }}">{{ chunk.ref_text }}</span>
                <span class="hyp {{ chunk.chunk_type }}">
                    <a href="javascript:;" onclick="
                        ws_mixed.seekTo({{ chunk.timestamp }}/ws_mixed.getDuration());
                        ws_mixed.play();
                        {% if html_vocals_fn %}
                        ws_vocals.seekTo({{ chunk.timestamp }}/ws_mixed.getDuration());
                        ws_vocals.play();
                        {% endif %}
                    ">{{ chunk.hyp_text }}</a>
                </span>
            </div>
        {% endfor %}
        </div>
    {% endfor %}
    <hr>
    </div>
{% endfor %}
</div>
</body>
</html>
"""

SUMMARY_HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>Transcription Summary</title>
  <style>
    body {
      font-family: 'Courier New', monospace;
      background-color: #f8f9fa;
      margin: 0;
      padding: 15px;
      line-height: 1.4;
    }

    h1 {
      color: #2c3e50;
      text-align: center;
      margin: 0 0 8px 0;
      font-size: 1.8em;
    }

    h2 {
      color: #34495e;
      margin: 8px 0;
      font-size: 1.2em;
    }

    .divider {
      padding: 0;
      margin: 12px 0;
    }
    .divider hr {
      margin: 0;
      border: none;
      border-top: 2px solid #ddd;
      clear: both;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      margin: 15px 0;
      box-shadow: 0 2px 10px rgba(0,0,0,0.1);
      background-color: white;
      border-radius: 8px;
      overflow: hidden;
    }
    th, td {
      padding: 10px 12px;
      text-align: left;
      border-bottom: 1px solid #e1e8ed;
      font-size: 0.95em;
    }
    th {
      background-color: #ecf0f1;
      color: #2c3e50;
      font-weight: bold;
    }
    tr:hover td {
      background-color: #f1f3f5;
    }

    a {
      color: #3498db;
      text-decoration: none;
      transition: color 0.2s ease;
    }
    a:hover {
      color: #2980b9;
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
      body {
        padding: 10px;
      }
      table, th, td {
        display: block;
        width: 100%;
      }
      th, td {
        box-sizing: border-box;
        width: 100%;
      }
      tr {
        margin-bottom: 10px;
        display: block;
      }
    }
  </style>
</head>
<body>
  <h1>Summary: {{ title }}</h1>
  <h2>Pipeline information: {{ description }}</h2>
  <h2>Headline WER: {{ summary_metrics['wer'] }}</h2>

  <table>
    <tr>
      <th>uid</th>
      {% for name in metric_order %}
        <th>{{ name }}</th>
      {% endfor %}
    </tr>
    <tr>
      <td><b>Totals</b></td>
      {% for name in metric_order %}
        <td>{{ "%.2f"|format(summary_metrics[name]) }}</td>
      {% endfor %}
    </tr>
    <tr><td class="divider" colspan="{{ num_columns }}"><hr/></td></tr>
    {% for ds, metrics in ds_metrics.items() %}
      <tr>
        <td><b>{{ ds }}</b></td>
        {% for name in metric_order %}
          <td>{{ "%.2f"|format(metrics[name]) }}</td>
        {% endfor %}
      </tr>
    {% endfor %}
    <tr><td class="divider" colspan="{{ num_columns }}"><hr/></td></tr>
    {% for _, song in df_songs.iterrows() %}
      <tr>
        <td>
          <a href="{{ song.uid }}.html">{{ song.uid }}</a>
        </td>
        <td>{{ "%.2f"|format(song.wer) }}</td>
        <td>
          {{ "%.2f"|format(
               song.subs / 
               (song.subs + song.dels + song.hits)
               if (song.subs + song.dels + song.hits)>0 else 0
             ) }}
        </td>
        <td>
          {{ "%.2f"|format(
               song.dels / 
               (song.subs + song.dels + song.hits)
               if (song.subs + song.dels + song.hits)>0 else 0
             ) }}
        </td>
        <td>
          {{ "%.2f"|format(
               song.inss / 
               (song.subs + song.dels + song.hits)
               if (song.subs + song.dels + song.hits)>0 else 0
             ) }}
        </td>
      </tr>
    {% endfor %}
  </table>
</body>
</html>
"""
