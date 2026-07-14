# make_annotation_page.py — 把 (keyframe, generated) 图对打包成单文件标注网页(图片内嵌base64, 双击即用)。
# 标完点"导出结果"下载 json。用法:
#   python3 pipeline/make_annotation_page.py --imgdir demo_2026-07-14/gen_e2_40/img --out demo_2026-07-14/标注页_忠实性人评.html
import os, json, base64, argparse, glob


def b64(p):
    return "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgdir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_n", type=int, default=50)
    a = ap.parse_args()
    kfs = sorted(glob.glob(os.path.join(a.imgdir, "*_keyframe.png")))
    pairs = []
    for kf in kfs:
        gen = kf.replace("_keyframe.png", "_generated.png")
        if os.path.exists(gen):
            vid = os.path.basename(kf).replace("_keyframe.png", "")
            pairs.append((vid, kf, gen))
    pairs = pairs[:a.max_n]
    print(f"打包 {len(pairs)} 对")
    items = [{"id": v, "kf": b64(k), "gen": b64(g)} for v, k, g in pairs]
    html = """<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>忠实性人评(%d对)</title><style>
body{font-family:-apple-system,'PingFang SC',sans-serif;margin:0;background:#f5f7fa;color:#1a202c}
.top{position:sticky;top:0;background:#fff;padding:10px 16px;box-shadow:0 1px 4px rgba(0,0,0,.08);z-index:9}
.top h3{margin:0 0 4px;font-size:16px}.top p{margin:0;font-size:12.5px;color:#4a5568}
.card{max-width:860px;margin:14px auto;background:#fff;border-radius:12px;padding:14px;box-shadow:0 1px 6px rgba(0,0,0,.06)}
.imgs{display:flex;gap:8px}.imgs div{flex:1;text-align:center}
.imgs img{width:100%%;border-radius:8px;border:1px solid #e2e8f0}
.imgs span{font-size:12px;color:#718096}
.btns{display:flex;gap:8px;margin-top:10px}
.btns button{flex:1;padding:12px 0;font-size:15px;border:none;border-radius:9px;cursor:pointer}
.yes{background:#c6f6d5}.no{background:#fed7d7}.unsure{background:#edf2f7}
.done{outline:3px solid #3182ce}
#bar{height:5px;background:#3182ce;width:0;transition:.2s}
#export{position:fixed;bottom:14px;right:14px;padding:12px 18px;font-size:14px;border:none;border-radius:10px;background:#3182ce;color:#fff;cursor:pointer;box-shadow:0 2px 8px rgba(0,0,0,.2)}
</style></head><body>
<div class="top"><div id="bar"></div><h3>忠实性人评 · 共 %d 对</h3>
<p>判断标准只有一条:<b>如果有人说右图是根据左图那一刻做的海报,你信不信?</b> 信=是;人物/场景/动作变了=不是;拿不准=说不清。不看画质美丑。</p></div>
<div id="list"></div>
<button id="export" onclick="exp()">导出结果(已标 <span id="cnt">0</span>/%d)</button>
<script>
const DATA=%s; const ans={};
const list=document.getElementById('list');
DATA.forEach((d,i)=>{
  const c=document.createElement('div'); c.className='card'; c.id='c'+i;
  c.innerHTML=`<div class="imgs"><div><img src="${d.kf}"><span>原始关键帧</span></div>
  <div><img src="${d.gen}"><span>AI 生成</span></div></div>
  <div class="btns"><button class="yes" onclick="mark(${i},'yes',this)">✅ 是同一瞬间</button>
  <button class="no" onclick="mark(${i},'no',this)">❌ 不是</button>
  <button class="unsure" onclick="mark(${i},'unsure',this)">🤔 说不清</button></div>`;
  list.appendChild(c);});
function mark(i,v,btn){ans[DATA[i].id]=v;
  btn.parentElement.querySelectorAll('button').forEach(b=>b.classList.remove('done'));
  btn.classList.add('done');
  const n=Object.keys(ans).length; document.getElementById('cnt').textContent=n;
  document.getElementById('bar').style.width=(100*n/DATA.length)+'%%';
  const nx=document.getElementById('c'+(i+1)); if(nx&&n<DATA.length) nx.scrollIntoView({behavior:'smooth'});}
function exp(){const blob=new Blob([JSON.stringify(ans,null,2)],{type:'application/json'});
  const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download='human_faithfulness_labels.json'; a.click();}
</script></body></html>""" % (len(items), len(items), len(items), json.dumps(items))
    open(a.out, "w", encoding="utf-8").write(html)
    print("标注页 ->", a.out, f"{os.path.getsize(a.out)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
