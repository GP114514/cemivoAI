"""Regenerate the retained generic neural narration, using voice-cues.json.

Requires edge-tts==7.3.1, srt==3.5.3 and FFmpeg. No API key is used.
"""
import asyncio
import json
import subprocess
from datetime import timedelta
from importlib.metadata import version
from pathlib import Path

import edge_tts
import srt

ROOT = Path(__file__).resolve().parent
AUDIO = ROOT / 'audio'
VOICE = 'en-US-AndrewNeural'
RATE = '-2%'
VOLUME = '+0%'


def probe(file):
    return float(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
        'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', str(file)], text=True))


async def main():
    config = json.loads((ROOT / 'voice-cues.json').read_text(encoding='utf-8-sig'))
    entries = config['cues']
    AUDIO.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(2)

    async def generate(index, cue):
        async with sem:
            at, text = cue['at'], cue['text']
            stem = AUDIO / f'voice-{index:02d}'
            mp3, wav, metadata = stem.with_suffix('.mp3'), stem.with_suffix('.wav'), stem.with_suffix('.jsonl')
            for attempt in range(3):
                try:
                    await edge_tts.Communicate(text, VOICE, rate=RATE, volume=VOLUME,
                        boundary='WordBoundary').save(str(mp3), str(metadata))
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    await asyncio.sleep((attempt + 1) * 2)
            submaker = edge_tts.SubMaker()
            for row in metadata.read_text(encoding='utf-8').splitlines():
                message = json.loads(row)
                if message.get('type') == 'WordBoundary':
                    submaker.feed(message)
            stem.with_suffix('.srt').write_text(submaker.get_srt(), encoding='utf-8')
            subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(mp3),
                '-ar', '48000', '-ac', '2', '-c:a', 'pcm_s16le', str(wav)], check=True)
            duration = probe(wav)
            next_at = entries[index + 1]['at'] if index + 1 < len(entries) else config['duration']
            if at + duration > next_at - .10:
                raise RuntimeError(f'Narration exceeds its slot: clip={index}, end={at + duration}, next={next_at}')
            print(json.dumps({'clip': index, 'duration': duration, 'end': round(at + duration, 3)}), flush=True)
            return {'at': at, 'text': text, 'file': f'audio/voice-{index:02d}.wav', 'duration': duration}, submaker.cues

    results = await asyncio.gather(*(generate(i, cue) for i, cue in enumerate(entries)))
    cues = [row[0] for row in results]
    (ROOT / 'voice-cues.json').write_text(json.dumps({'duration': config['duration'], 'cues': cues}, indent=2), encoding='utf-8')
    info = {'voice': VOICE, 'rate': RATE, 'volume': VOLUME, 'offline': False, 'online': True,
        'service': 'Microsoft Edge online neural TTS', 'engine': 'edge-tts', 'version': version('edge-tts'),
        'generic_synthesized_voice': True, 'voice_cloning': False, 'api_key_used': False,
        'clips': len(cues), 'output_sample_rate': 48000, 'channels': 2}
    (AUDIO / 'voice-info.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    words = []
    for cue, entries_words in results:
        for word in entries_words:
            words.append(srt.Subtitle(index=len(words)+1, start=word.start+timedelta(seconds=cue['at']),
                end=word.end+timedelta(seconds=cue['at']), content=word.content))
    (AUDIO / 'narration-words.srt').write_text(srt.compose(words), encoding='utf-8')
    (AUDIO / 'narration-captions.srt').write_text(srt.compose([srt.Subtitle(index=i+1,
        start=timedelta(seconds=c['at']), end=timedelta(seconds=c['at']+c['duration']), content=c['text'])
        for i, c in enumerate(cues)]), encoding='utf-8')
    print(json.dumps({'complete': True, 'last_end': cues[-1]['at']+cues[-1]['duration']}))


if __name__ == '__main__':
    asyncio.run(main())
