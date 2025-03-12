import argparse
from typing import Dict, List

import plotly.graph_objs as go
from plotly.subplots import make_subplots

from pyannote.core import Annotation
from pyannote.database.util import load_rttm


def as_dict_list(annotation: Annotation) -> Dict[str, List[Dict]]:
    result = {label: [] for label in annotation.labels()}
    for segment, track, label in annotation.itertracks(yield_label=True):
        result[label].append({
            "speaker": label,
            "start": segment.start,
            "end": segment.end,
            "duration": segment.duration,
            "track": track,
        })
    return result


def plot_annotation(annotation: Annotation,fig,colors,show, r, c):
    data = as_dict_list(annotation)
    
    for label, turns in data.items():
        print(label)
        durations, starts, ends = [], [], []
        for turn in turns:
            durations.append(turn["duration"])
            starts.append(turn["start"])
            ends.append(f"{turn['end']:.1f}")
        fig.add_bar(
            x=durations,
            y=[label] * len(durations),
            base=starts,
            orientation='h',
            showlegend=show,
            marker_color=colors[label],
            name=label,
            hovertemplate="<b>%{base:.2f} --> %{x:.2f}</b>",
            row=r, col=c
        )

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-r', dest='reference', help='reference RTTM file')
    parser.add_argument('-s', dest='hypothesis', help='system RTTM file(s)', nargs='+')
    args = parser.parse_args()
    args.reference, args.hypothesis
    rttm_ref = load_rttm(args.reference)
    
    #rttm_hyp = load_rttm(args.hypothesis)
    
    fig = make_subplots(rows=5, subplot_titles=('RTTM_ref','simple_diarizer_pyannote_vad' , 'simple_diarizer_speechbrain_vad' , 'simple_diarizer_silero_vad','simple_diarizer_auditok_vad' ))
    colors = {'spk1':'steelblue', "spk2": 'green', "spk3": 'indigo', "spk4": 'orange',"4": 'cyan',"5":'darkslategray', 
             "da":'greenyellow',"et":'hotpink',"fi":'indigo', "el":'khaki', "hu":'lightblue', "lt":'lightgreen',
               "mt":'lightpink', "nl":'magenta', "pl":'maroon', "pt":'navy', 
             "ro":'olive', "sk":'orange', "sl":'pink', "sv":'salmon',"cs":'silver',
          'fr':'firebrick'}
    for uri, annotation in rttm_ref.items():
        plot_annotation(annotation,fig,colors,True, 1, 1)
    R=2  
    for hyp_rttm_fn in args.hypothesis:
        print(hyp_rttm_fn)
        rttm_hyp = load_rttm(hyp_rttm_fn)
        
        for uri, annotation in rttm_hyp.items():
            plot_annotation(annotation,fig,colors,False, R, 1)
        R=R+1
        
    fig.update_layout(
        title=annotation.uri,
        legend_title="Speakers",
        
        
        
    )      
    fig.show()
     