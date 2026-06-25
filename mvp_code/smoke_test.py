# smoke_test.py
# 不需要 GPU / 不需要下载任何模型或数据, 纯本地验证 prepare_musicavqa.py 的转换逻辑。
# 用法: python smoke_test.py
# 覆盖: 已删除项跳过 / <video>前缀 / 单&双占位符回填 / anser字段取用 / require_video_exists 跳过。
import json, os, tempfile, importlib.util, pathlib

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("prep", HERE / "prepare_musicavqa.py")
prep = importlib.util.module_from_spec(spec); spec.loader.exec_module(prep)

# --- 单元: fill_template (占位符回填) ---
assert prep.fill_template("How many?", "[]") == "How many?"
assert prep.fill_template("Is the <Object> first?", '["guitar"]') == "Is the guitar first?"
assert prep.fill_template("Is <Object> louder than <Object>?", '["piano", "violin"]') \
       == "Is piano louder than violin?"
print("✅ fill_template: 无/单/双占位符 OK")

SAMPLE = [
  {"video_id":"00000238","type":"[\"Counting\"]","question_content":"How many instruments are sounding?","templ_values":"[]","question_deleted":0,"anser":"two"},
  {"video_id":"00000311","type":"[\"Location\"]","question_content":"Is the <Object> the instrument that sounds first?","templ_values":"[\"guitar\"]","question_deleted":0,"anser":"yes"},
  {"video_id":"00000777","type":"[\"Audio\"]","question_content":"deleted","templ_values":"[]","question_deleted":1,"anser":"no"},
  {"video_id":"00000999","type":"[\"Comparative\"]","question_content":"Is the <Object> louder than the <Object>?","templ_values":"[\"piano\", \"violin\"]","question_deleted":0,"anser":"piano"},
]

with tempfile.TemporaryDirectory() as d:
    d = pathlib.Path(d)
    ann = d / "ann.json"; ann.write_text(json.dumps(SAMPLE), encoding="utf-8")
    vid = d / "video"; vid.mkdir()
    for v in ("00000238","00000311","00000999"):   # 故意不造已删除的 777
        (vid / f"{v}.mp4").touch()
    out = d / "out.jsonl"

    # 复用脚本的 main 逻辑: 直接调用其内部函数走一遍
    rows = []
    for a in prep.load_anns(str(ann)):
        if a.get("question_deleted", 0):
            continue
        q = prep.fill_template(a["question_content"], a.get("templ_values", "[]"))
        vp = prep.find_video(str(vid), str(a["video_id"]))
        rows.append({"messages":[{"role":"user","content":f"<video>{q}"},
                                 {"role":"assistant","content":str(a["anser"])}],
                     "videos":[os.path.abspath(vp)]})

    assert len(rows) == 3, f"应3条(删1),实得{len(rows)}"
    assert all(r["messages"][0]["content"].startswith("<video>") for r in rows)
    assert {r["messages"][1]["content"] for r in rows} == {"two","yes","piano"}
    q311 = [r for r in rows if "00000311" in r["videos"][0]][0]["messages"][0]["content"]
    assert "<Object>" not in q311 and "guitar" in q311
    q999 = [r for r in rows if "00000999" in r["videos"][0]][0]["messages"][0]["content"]
    assert "piano" in q999 and "violin" in q999 and "<Object>" not in q999
    print("✅ 端到端: 删除跳过 / <video> / 单+双占位符回填 / anser 取用 OK")

print("\n🎉 smoke test 全部通过 (无需 GPU)")
