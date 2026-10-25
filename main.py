"""Offline audio enhancement and source-grounded media editorial assistant."""
import argparse, json, wave, re
from pathlib import Path
import numpy as np
from common import save_report, table, lines

def enhance(audio, rate):
    # Estimate stationary noise from a declared leading noise-only interval.
    n=512; hop=128; padded=np.pad(audio,(n,n)); window=np.hanning(n)
    frames=np.array([padded[i:i+n]*window for i in range(0,len(padded)-n+1,hop)])
    spectrum=np.fft.rfft(frames); noise=np.mean(abs(spectrum[4:max(5,int(.20*rate/hop))])**2,axis=0)
    power=abs(spectrum)**2; gain=np.maximum(.08,1-noise/np.maximum(power,1e-12))
    recovered=np.fft.irfft(spectrum*gain,n=n); out=np.zeros(len(padded)); weight=np.zeros(len(padded))
    for i,frame in enumerate(recovered):
        start=i*hop; out[start:start+n]+=frame*window; weight[start:start+n]+=window**2
    return (out/np.maximum(weight,1e-9))[n:n+len(audio)]

def snr(clean,test):
    return float(10*np.log10(np.sum(clean**2)/max(np.sum((clean-test)**2),1e-12)))

def write_wav(path,audio,rate):
    with wave.open(str(path),'wb') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(rate); f.writeframes((np.clip(audio,-1,1)*32767).astype('<i2').tobytes())

def read_wav(path):
    with wave.open(str(path),'rb') as f:
        if f.getsampwidth()!=2: raise ValueError('Expected 16-bit PCM WAV; convert the input first.')
        rate=f.getframerate(); channels=f.getnchannels(); data=np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').astype(float)/32768
    return data.reshape(-1,channels).mean(axis=1),rate

def grounded_brief(segments, query):
    words=set(re.findall(r'\w+',query.lower())); ranked=sorted(segments,key=lambda s:len(words & set(re.findall(r'\w+',s['text'].lower()))),reverse=True)
    relevant=[s for s in ranked if words & set(re.findall(r'\w+',s['text'].lower()))][:3]
    return [{'start':s['start'],'end':s['end'],'quote':s['text']} for s in relevant]

def frame_quality(paths):
    # Optional video workflow: analyze exported PNG/JPG frames with Pillow.
    from PIL import Image
    rows=[]
    for path in paths:
        image=np.asarray(Image.open(path).convert('L'),dtype=float)
        if min(image.shape)<3: raise ValueError('Frame must be at least 3x3 pixels')
        lap=-4*image[1:-1,1:-1]+image[:-2,1:-1]+image[2:,1:-1]+image[1:-1,:-2]+image[1:-1,2:]
        rows.append((str(path),round(float(image.mean()),1),round(float(np.mean(image<10)),3),round(float(lap.var()),1)))
    return rows

def run(out,wav=None,transcript=None,frames=None):
    out=Path(out); out.mkdir(parents=True,exist_ok=True); rate=16000; rng=np.random.default_rng(9)
    t=np.arange(rate*4)/rate; clean=.25*np.sin(2*np.pi*220*t)+.12*np.sin(2*np.pi*440*t); clean[t<.4]=0
    if wav: noisy,rate=read_wav(wav); clean=None
    else: noisy=clean+rng.normal(0,.12,len(t))
    processed=enhance(noisy,rate); write_wav(out/'input.wav',noisy,rate); write_wav(out/'enhanced.wav',processed,rate)
    segments=json.loads(Path(transcript).read_text()) if transcript else [{'start':0,'end':8,'text':'Audience research found that short explainers receive more repeat views.'},{'start':8,'end':16,'text':'The team will test personalized recommendations in the next campaign.'},{'start':16,'end':24,'text':'Success will be measured using watch time and return visits.'}]
    for s in segments:
        if not isinstance(s.get('text'),str) or s['start']<0 or s['end']<=s['start']: raise ValueError('Invalid transcript segment')
    brief=grounded_brief(segments,'personalized recommendations campaign'); (out/'grounded_brief.json').write_text(json.dumps(brief,indent=2))
    metrics={'Audio seconds':round(len(noisy)/rate,2),'Clipping fraction':round(float(np.mean(abs(noisy)>=.999)),4),'Cited excerpts':len(brief),'Transcript mode':'provided' if transcript else 'independent sample fixture'}
    if clean is not None: metrics.update({'Input SNR dB':round(snr(clean,noisy),2),'Enhanced SNR dB':round(snr(clean,processed),2)})
    sections=[('Waveform preview',lines({'input':noisy[::max(1,len(noisy)//400)],'enhanced':processed[::max(1,len(processed)//400)]})),('Grounded editorial evidence',table(['start seconds','end seconds','source excerpt'],[(s['start'],s['end'],s['quote']) for s in brief])),('Scope','<p>The audio benchmark uses synthetic tones with known clean reference, not speech. The sample transcript is independent of this audio. The default editorial assistant retrieves verbatim evidence; it does not claim to run an LLM or ASR. A deployment prompt and ASR input contract are included separately. Real-input SNR cannot be measured without a clean reference. Leading 0.4 seconds should contain noise only.</p>')]
    if frames: sections.append(('Video frame quality proxies',table(['frame','mean brightness','dark fraction','Laplacian variance'],frame_quality(frames))))
    save_report(out,'Media Quality Lab',metrics,sections); return metrics

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',default='results'); p.add_argument('--wav'); p.add_argument('--transcript'); p.add_argument('--frames',nargs='*'); a=p.parse_args(); print(run(a.out,a.wav,a.transcript,a.frames))
