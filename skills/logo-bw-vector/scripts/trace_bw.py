#!/usr/bin/env python3
"""Trace verified black-on-white artwork to standalone vector paths.
Dependencies: Pillow and NumPy. Does not repair images or recognize text.
"""
import argparse
from pathlib import Path
from xml.etree import ElementTree as ET
import numpy as np
from PIL import Image

def rdp(p,eps):
 if len(p)<3:return np.array(p).tolist()
 p=np.array(p,float); v=p[-1]-p[0]; l=np.linalg.norm(v)
 d=np.abs(v[0]*(p-p[0])[:,1]-v[1]*(p-p[0])[:,0])/l if l else np.linalg.norm(p-p[0],axis=1)
 i=int(np.argmax(d))
 if d[i]>eps:return rdp(p[:i+1],eps)[:-1]+rdp(p[i:],eps)
 return [p[0].tolist(),p[-1].tolist()]
def paths(mask,eps=.5,minarea=3):
 m=np.pad(mask.astype(bool),1); edges={}
 def add(a,b):edges.setdefault(a,[]).append(b)
 for y,x in zip(*np.nonzero(m[1:-1,1:-1]&~m[:-2,1:-1])): add((int(x),int(y)),(int(x+1),int(y)))
 for y,x in zip(*np.nonzero(m[1:-1,1:-1]&~m[1:-1,2:])): add((int(x+1),int(y)),(int(x+1),int(y+1)))
 for y,x in zip(*np.nonzero(m[1:-1,1:-1]&~m[2:,1:-1])): add((int(x+1),int(y+1)),(int(x),int(y+1)))
 for y,x in zip(*np.nonzero(m[1:-1,1:-1]&~m[1:-1,:-2])): add((int(x),int(y+1)),(int(x),int(y)))
 result=[]
 while edges:
  start=next(iter(edges));p=[start];prev=start;q=edges[start].pop()
  if not edges[start]:del edges[start]
  while q!=start:
   p.append(q)
   if q not in edges:raise RuntimeError('Open contour')
   options=edges[q];vx=q[0]-prev[0];vy=q[1]-prev[1]
   def priority(z):
    dx=z[0]-q[0];dy=z[1]-q[1];cross=vx*dy-vy*dx;dot=vx*dx+vy*dy
    return 3 if cross>0 else 2 if dot>0 else 1 if cross<0 else 0
   nxt=max(options,key=priority);options.remove(nxt)
   if not options:del edges[q]
   prev=q;q=nxt
  if len(p)<4:continue
  area=abs(sum(p[i][0]*p[(i+1)%len(p)][1]-p[(i+1)%len(p)][0]*p[i][1] for i in range(len(p)))/2)
  if area<minarea:continue
  k=len(p)//2;s=rdp(p[:k+1],eps)[:-1]+rdp(p[k:]+[p[0]],eps)[:-1]
  if len(s)<3:continue
  n=len(s);ins=[];outs=[]
  for i in range(n):
   cur=np.array(s[i]);prev=np.array(s[(i-1)%n]);nxt=np.array(s[(i+1)%n])
   dl=np.linalg.norm(prev-cur);dr=np.linalg.norm(nxt-cur)
   radius=min(eps*.45,dl*.1,dr*.1)
   ins.append(cur+(prev-cur)*radius/dl if dl else cur)
   outs.append(cur+(nxt-cur)*radius/dr if dr else cur)
  d=f'M {ins[0][0]:.2f},{ins[0][1]:.2f}'
  for i in range(n):
   if i:d+=f' L {ins[i][0]:.2f},{ins[i][1]:.2f}'
   d+=f' Q {s[i][0]:.2f},{s[i][1]:.2f} {outs[i][0]:.2f},{outs[i][1]:.2f}'
  result.append(d+' Z')
 return ' '.join(result)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--threshold", type=int, default=145)
    parser.add_argument("--simplify", type=float, default=0.65)
    parser.add_argument("--min-area", type=float, default=3)
    parser.add_argument("--mask-output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.threshold <= 255 or args.simplify < 0 or args.min_area < 0:
        parser.error("threshold must be 1..255; simplify and min-area must be nonnegative")
    if args.input.resolve() == args.output.resolve():
        parser.error("output must differ from the source image")
    if args.mask_output and args.input.resolve() == args.mask_output.resolve():
        parser.error("mask output must differ from the source image")
    rgba = Image.open(args.input).convert("RGBA")
    rgb = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")
    mask = np.array(rgb.convert("L")) < args.threshold
    d = paths(mask, args.simplify, args.min_area)
    if not d:
        raise ValueError("No contours found; check source polarity and threshold")
    w, h = rgb.size
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
           f'viewBox="0 0 {w} {h}"><rect width="100%" height="100%" fill="white"/>'
           f'<path d="{d}" fill="black" fill-rule="evenodd"/></svg>')
    ET.fromstring(svg)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")
    if args.mask_output:
        args.mask_output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.where(mask, 0, 255).astype("uint8")).save(args.mask_output)
    print(f"Saved {args.output} ({w}x{h}, vector paths, no embedded raster)")

if __name__ == "__main__":
    main()
