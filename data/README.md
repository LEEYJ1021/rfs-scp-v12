# Data

## AnnoMI Dataset

This directory is a placeholder for the AnnoMI dataset files.

**The CSV files are NOT tracked in this repository.**

### Download

1. Go to: https://github.com/uccollab/annomi
2. Download `AnnoMI-full.csv` and/or `AnnoMI-simple.csv`
3. Place them here: `data/annomi/AnnoMI-full.csv`

### Expected Schema (AnnoMI-full.csv)

| Column | Type | Description |
|--------|------|-------------|
| mi_quality | str | `high` or `low` — expert MI quality label |
| transcript_id | int | Session identifier (0–132) |
| video_title | str | Source YouTube video title |
| video_url | str | Source URL |
| topic | str | Behaviour change topic (44 unique) |
| utterance_id | int | Utterance order within session |
| interlocutor | str | `therapist` or `client` |
| timestamp | str | MM:SS position in video |
| utterance_text | str | Raw transcript text |
| annotator_id | int | Annotator identifier (0–9) |
| therapist_input_exists | bool/NaN | Therapist input behaviour present |
| therapist_input_subtype | str/NaN | information / advice / negotiation / options |
| reflection_exists | bool/NaN | Reflection behaviour present |
| reflection_subtype | str/NaN | simple / complex |
| question_exists | bool/NaN | Question behaviour present |
| question_subtype | str/NaN | open / closed |
| main_therapist_behaviour | str/NaN | other / question / reflection / therapist_input |
| client_talk_type | str/NaN | neutral / change / sustain |

### Dataset Statistics

- **Total utterances**: 13,551
- **Sessions**: 133
- **MI quality**: High=12,286 utterances / Low=1,265 utterances (utterance-level); High=110 sessions / Low=23 sessions (session-level)
- **Mean utterances/session**: 101.9 (median=50, min=6, max=1,750)
- **Topics**: 44 unique (alcohol, smoking, exercise, diabetes, etc.)

### Citation

```bibtex
@article{wu2022annomi,
  author  = {Wu, Zixiu and others},
  title   = {{AnnoMI}: A Dataset of Expert-Annotated Counselling Dialogues},
  year    = {2022}
}
```
