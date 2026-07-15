# make_dual_annotation_page.py — 双轴标注页: 每对图两问(①同一瞬间? ②海报感/艺术性?)
# 用法: python3 pipeline/make_dual_annotation_page.py --imgdir <dir> --out <html> [--max_n 10]
import os, json, base64, argparse, glob


def b64(p):
    return "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgdir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_n", type=int, default=10)
    a = ap.parse_args()
    kfs = sorted(glob.glob(os.path.join(a.imgdir, "*_keyframe.png")))
    pairs = []
    for kf in kfs:
        gen = kf.replace("_keyframe.png", "_generated.png")
        if os.path.exists(gen):
            pairs.append((os.path.basename(kf).replace("_keyframe.png", ""), kf, gen))
    pairs = pairs[:a.max_n]
    items = [{"id": v, "kf": b64(k), "gen": b64(g)} for v, k, g in pairs]
    html = """<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>双轴人评(%d对)</title><style>
body{font-family:-apple-system,'PingFang SC',sans-serif;margin:0;background:#f5f7fa;color:#1a202c}
.top{position:sticky;top:0;background:#fff;padding:10px 16px;box-shadow:0 1px 4px rgba(0,0,0,.08);z-index:9}
.top h3{margin:0 0 4px;font-size:16px}.top p{margin:0;font-size:12px;color:#4a5568}
.card{max-width:860px;margin:14px auto;background:#fff;border-radius:12px;padding:14px;box-shadow:0 1px 6px rgba(0,0,0,.06)}
.imgs{display:flex;gap:8px}.imgs div{flex:1;text-align:center}
.imgs img{width:100%%;border-radius:8px;border:1px solid #e2e8f0}
.imgs span{font-size:12px;color:#718096}
.q{margin-top:10px;font-size:13.5px;font-weight:600}
.btns{display:flex;gap:8px;margin-top:6px}
.btns button{flex:1;padding:10px 0;font-size:14px;border:none;border-radius:9px;cursor:pointer}
.yes{background:#c6f6d5}.no{background:#fed7d7}.unsure{background:#edf2f7}
.a1{background:#fed7d7}.a2{background:#feebc8}.a3{background:#c6f6d5}
.done{outline:3px solid #3182ce}
#bar{height:5px;background:#3182ce;width:0;transition:.2s}
#export{position:fixed;bottom:14px;right:14px;padding:12px 18px;font-size:14px;border:none;border-radius:10px;background:#3182ce;color:#fff;cursor:pointer;box-shadow:0 2px 8px rgba(0,0,0,.2)}
</style></head><body>
<div class="top"><div id="bar"></div><h3>双轴人评 · 共 %d 对 × 2 问</h3>
<p>Q1 忠实: 右图是左图那一刻的海报,信不信?(人物/衣着/动作/场景一致; 光影美化允许)。Q2 海报感: 右图作为宣传海报的艺术质感——1=就是截图/像素增强, 2=有些质感提升, 3=像正经海报。</p></div>
<div id="list"></div>
<button id="export" onclick="exp()">导出(已标 <span id="cnt">0</span>/%d)</button>
<script>
const DATA=%s; const ans={};
const list=document.getElementById('list');
DATA.forEach((d,i)=>{
  ans[d.id]={};
  const c=document.createElement('div'); c.className='card'; c.id='c'+i;
  c.innerHTML=`<div class="imgs"><div><img src="${d.kf}"><span>原始关键帧</span></div>
  <div><img src="${d.gen}"><span>AI 生成</span></div></div>
  <div class="q">Q1 是同一瞬间吗?</div>
  <div class="btns"><button class="yes" onclick="mark(${i},'faithful','yes',this)">✅ 是</button>
  <button class="no" onclick="mark(${i},'faithful','no',this)">❌ 不是</button>
  <button class="unsure" onclick="mark(${i},'faithful','unsure',this)">🤔 说不清</button></div>
  <div class="q">Q2 海报感(艺术质感)?</div>
  <div class="btns"><button class="a1" onclick="mark(${i},'artistry','1',this)">1 像截图</button>
  <button class="a2" onclick="mark(${i},'artistry','2',this)">2 有质感</button>
  <button class="a3" onclick="mark(${i},'artistry','3',this)">3 像海报</button></div>`;
  list.appendChild(c);});
function total(){let n=0; for(const k in ans){if(ans[k].faithful&&ans[k].artistry)n++;} return n;}
function mark(i,key,v,btn){ans[DATA[i].id][key]=v;
  btn.parentElement.querySelectorAll('button').forEach(b=>b.classList.remove('done'));
  btn.classList.add('done');
  const n=total(); document.getElementById('cnt').textContent=n;
  document.getElementById('bar').style.width=(100*n/DATA.length)+'%%';
  if(ans[DATA[i].id].faithful&&ans[DATA[i].id].artistry&&n<DATA.length){
    const nx=document.getElementById('c'+(i+1)); if(nx) nx.scrollIntoView({behavior:'smooth'});}}
function exp(){const blob=new Blob([JSON.stringify(ans,null,2)],{type:'application/json'});
  const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download='dual_axis_labels.json'; a.click();}
</script></body></html>""" % (len(items), len(items), len(items), json.dumps(items))
    open(a.out, "w", encoding="utf-8").write(html)
    print("双轴标注页 ->", a.out, f"{os.path.getsize(a.out)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
