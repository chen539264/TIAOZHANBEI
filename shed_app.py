"""Electric bicycle shed video demonstration with simulated SMS."""
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from datetime import datetime

import streamlit as st
from shed_events import Store, Gate

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='棚安 · 车棚火情预警',page_icon='🚲',layout='wide')
st.markdown('''<style>
.stApp {background:#f3f6fa;color:#172b44;}
.block-container {max-width:1280px;padding-top:2rem;padding-bottom:3rem;}
h1,h2,h3 {letter-spacing:-.03em;color:#172b44;}
.hero {position:relative;overflow:hidden;padding:32px 36px;border-radius:22px;background:linear-gradient(115deg,#102d43,#174e5b);color:white;margin-bottom:18px;box-shadow:0 12px 30px #14394b15;}
.hero::after {content:'◎';position:absolute;right:40px;top:-85px;font-size:300px;color:#ffffff08;pointer-events:none;}
.eyebrow {font-size:12px;letter-spacing:.18em;color:#83d8d2;font-weight:700;}
.hero h1 {color:white!important;font-size:32px!important;margin:12px 0 8px!important;padding:0!important;}
.hero p {color:#c4d9e3;max-width:720px;margin:0;line-height:1.8;}
.pills {display:flex;gap:8px;flex-wrap:wrap;margin-top:22px;}
.pill {font-size:12px;background:#ffffff14;border:1px solid #ffffff26;padding:5px 12px;border-radius:30px;color:#e6f5f5;}
.notice {padding:12px 18px;background:#fff7e7;border:1px solid #f0dfbc;border-radius:12px;color:#795619;font-size:14px;margin-bottom:22px;}
[data-testid="stMetric"] {background:white;border:1px solid #e3eaf1;border-radius:16px;padding:18px 22px;box-shadow:0 3px 10px #14394b04;}
[data-testid="stMetricValue"] {color:#176e77;font-weight:700;}
[data-baseweb="tab-list"] {gap:16px;background:transparent;border-bottom:1px solid #dce5ee;margin:24px 0 18px;}
[data-baseweb="tab"] {height:52px;font-weight:600;font-size:15px;padding:0 12px;}
[data-baseweb="tab"][aria-selected="true"] {color:#087e83;}
[data-testid="stFileUploader"] {background:#fff;border:1px dashed #abc4d3;border-radius:16px;padding:16px;}
[data-testid="stExpander"] {background:#fff;border:1px solid #e1e8ef;border-radius:14px;}
.stButton>button,.stDownloadButton>button {border-radius:10px;min-height:42px;}
.stButton>button[kind="primary"] {background:#087e83;border-color:#087e83;}
.section-kicker {color:#638299;letter-spacing:.1em;font-size:11px;font-weight:700;margin:12px 0 4px;}
.empty {text-align:center;background:linear-gradient(135deg,#eaf1f6,#f9fbfd);border:1px solid #dce7ef;border-radius:18px;padding:40px 20px;margin:14px 0;color:#627f92;}
.empty strong {display:block;color:#294b61;font-size:19px;margin:10px 0;}
.flow {display:flex;gap:8px;flex-wrap:wrap;margin:10px 0 20px;}
.flow span {background:#e7f2f2;color:#28666d;border-radius:8px;padding:7px 12px;font-size:12px;}
@media(max-width:640px){.hero{padding:24px}.hero h1{font-size:25px!important}.block-container{padding-top:1rem}.hero::after{display:none}}
</style>
<div class="hero"><div class="eyebrow">CAMPUS SAFETY / 校园安全</div>
<h1>棚安 · 车棚火情预警</h1><p>让每一次异常有迹可循。连接车棚画面与保卫人员，记录从发现、接警到处置的全过程。</p>
<div class="pills"><span class="pill">浙江万里学院</span><span class="pill">视频演示模式</span><span class="pill">短信通道 · 模拟</span></div></div>
<div class="notice">● 演示环境：视频文件模拟监控，短信仅生成预览，不会发送到手机。</div>''',unsafe_allow_html=True)
store=Store(Path(os.environ.get('SHED_DATA_DIR',str(ROOT/'shed_runtime'))))
overview=store.all()
summary_cols=st.columns(4)
summary_cols[0].metric('累计告警事件',len(overview))
summary_cols[1].metric('等待接警',sum(x['status']=='待接警' for x in overview))
summary_cols[2].metric('正在处置',sum(x['status']=='处理中' for x in overview))
summary_cols[3].metric('已完成处置',sum(x['status']=='已结案' for x in overview))

@st.cache_resource
def model_resource(path, revision):
    import threading
    from ultralytics import YOLO
    import torch
    torch.set_num_threads(4)
    return YOLO(path), threading.Lock()

def find_weight():
    candidates=[ROOT/'artifacts/best.pt',ROOT/'best.pt',ROOT.parent/'best.pt']
    return next((p for p in candidates if p.is_file()),candidates[0])

def predict(frame,threshold):
    p=find_weight()
    if not p.exists():
        raise ValueError('请将模型放在 artifacts/best.pt')
    model,lock=model_resource(str(p),p.stat().st_mtime_ns)
    classes=[k for k,v in model.names.items() if v.lower()=='fire']
    if not classes:
        raise ValueError('模型中没有 fire 类别')
    with lock:
        r=model.predict(frame,conf=threshold,classes=classes,imgsz=960,device='cpu',verbose=False)[0]
        score=float(r.boxes.conf.max()) if len(r.boxes) else 0.0
        return score, r.plot()

video_tab,desk,image_tab=st.tabs(['01  视频监测','02  接警与处置','03  图片检测'])
with video_tab:
    st.markdown('<div class="section-kicker">VIDEO MONITORING</div>',unsafe_allow_html=True)
    st.subheader('从一段车棚视频开始')
    st.caption('检测按采样帧运行，不是实时摄像头；连续命中用于演示确认，未通过车棚场景校准。')
    a,b=st.columns(2)
    site=a.text_input('车棚位置','浙江万里学院 · 示例车棚')
    contact=b.text_input('演示值班人员','保卫部测试值班员')
    with st.expander('检测参数 · 默认使用现有模型配置',expanded=False):
        a,b,c=st.columns(3)
        threshold=a.number_input('检测阈值',0.001,0.9999,0.1,format='%.4f')
        interval=b.selectbox('采样间隔（视频秒）',[0.5,1.0,2.0],index=1)
        required=c.number_input('连续命中采样帧数',1,10,3)
        st.caption('连续 5 个采样帧未检出后，才允许为新的火情生成事件。重复运行同一视频会作为新的演示批次。')
    st.markdown('<div class="flow"><span>01 上传视频</span><span>02 识别异常</span><span>03 生成模拟短信</span><span>04 人员接警处置</span></div>',unsafe_allow_html=True)
    upload=st.file_uploader('上传视频（建议短视频，最多分析前 10 分钟）',type=['mp4','avi','mov','mkv'])
    if upload:
        st.video(upload)
    else:
        st.markdown('<div class="empty">▷<strong>监测画面待接入</strong>上传一段车棚视频，查看疑似火情与事件记录<br><small>建议使用短视频 · 最多分析前 10 分钟</small></div>',unsafe_allow_html=True)
    if st.button('开始视频分析',disabled=upload is None,type='primary'):
        if not site.strip() or not contact.strip():
            st.error('请填写车棚位置与值班人员')
        else:
            cap=None; temp=None
            try:
                import cv2
                blob=upload.getvalue()
                with tempfile.NamedTemporaryFile(suffix=Path(upload.name).suffix,delete=False) as f:
                    f.write(blob);temp=f.name
                cap=cv2.VideoCapture(temp)
                fps=cap.get(cv2.CAP_PROP_FPS)
                if not cap.isOpened() or not 0<fps<1000:
                    raise ValueError('无法解码视频或帧率无效，请转换为标准 MP4 后重试')
                gate=Gate(int(required));stride=max(1,round(fps*interval))
                preview=st.empty();status=st.empty();start=time.perf_counter()
                samples=0;events=[];idx=0;inference=0.;ended=False
                while idx/fps<600:
                    ok,frame=cap.read()
                    if not ok:
                        ended=True;break
                    t=idx/fps;idx+=1
                    if (idx-1)%stride:continue
                    tick=time.perf_counter();score,marked=predict(frame,threshold)
                    inference+=time.perf_counter()-tick;samples+=1
                    first=gate.step(score>=threshold,t)
                    preview.image(marked,channels='BGR',caption=f'视频 {t:.1f}s · '+('疑似火情' if score>=threshold else '本采样帧未检出'),use_container_width=True)
                    if first is not None:
                        # Store evidence before publishing an actionable event.
                        import uuid
                        evidence=uuid.uuid4().hex+'.jpg'
                        if not cv2.imwrite(str(store.root/evidence),marked):
                            raise ValueError('告警截图保存失败')
                        eid=store.create(site=site,contact=contact,video=upload.name,video_sha256=hashlib.sha256(blob).hexdigest(),
                            video_time=t,first_positive_time=first,confirmation_video_seconds=t-first,
                            analysis_elapsed_seconds=time.perf_counter()-start,threshold=threshold,score=score,evidence=evidence)
                        events.append(eid)
                    status.info(f'已分析 {samples} 个采样帧，生成 {len(events)} 个事件 · 模拟短信 {len(events)} 条')
                if samples==0:raise ValueError('视频未解码出任何有效帧')
                result=dict(video=upload.name,samples=samples,event_ids=events,elapsed_seconds=round(time.perf_counter()-start,2),
                    mean_inference_seconds=round(inference/samples,3),coverage='解码器返回视频结束（损坏文件也可能提前结束）' if ended else '仅分析前10分钟',sms_mode='模拟发送')
                st.session_state['last_analysis']=result
                st.success('本轮分析完成，请进入“接警与处置”处理事件。')
            except Exception as e:
                st.error(f'视频分析失败：{e}。此前已生成的事件仍保留在接警台。')
            finally:
                if cap is not None:cap.release()
                if temp:Path(temp).unlink(missing_ok=True)
    if 'last_analysis' in st.session_state:
        result=st.session_state['last_analysis']
        st.subheader('本轮分析概览')
        cols=st.columns(3)
        cols[0].metric('采样帧数',result['samples'])
        cols[1].metric('生成事件',len(result['event_ids']))
        cols[2].metric('分析用时',f"{result['elapsed_seconds']:.1f} 秒")
        st.caption(result['coverage'])
        with st.expander('查看完整分析记录'):st.json(result)

with desk:
    st.markdown('<div class="section-kicker">INCIDENT RESPONSE</div>',unsafe_allow_html=True)
    st.subheader('保卫部接警台')
    st.caption('此页为同一应用内的演示接警台，尚未实现手机短信链接、身份认证或自动升级通知。共享部署时，所有访问者均可操作演示事件。')
    if st.button('刷新事件'):st.rerun()
    rows=store.all()
    a,b,c=st.columns(3)
    a.metric('待接警',sum(r['status']=='待接警' for r in rows))
    b.metric('处理中',sum(r['status']=='处理中' for r in rows))
    c.metric('已结案',sum(r['status']=='已结案' for r in rows))
    st.caption('人工判为误报的事件数：'+str(sum(r.get('verdict')=='误报' for r in rows))+'。缺少完整真实标注，暂不计算误报率或漏检率。')
    st.download_button('导出事件与处置记录',json.dumps(rows,ensure_ascii=False,indent=2),'shed_events.json','application/json')
    filter_status=st.selectbox('查看事件',['全部','待接警','处理中','已结案'])
    rows=[r for r in rows if filter_status=='全部' or r['status']==filter_status]
    if not rows:st.info('当前没有符合条件的事件。视频分析生成的告警会出现在这里。')
    for r in rows:
        badge={'待接警':'🔴','处理中':'🟠','已结案':'🟢'}[r['status']]
        with st.expander(f"{badge} {r['status']} ｜ {r['site']} ｜ {r['id'][:8]}",expanded=False):
            st.write(f"值班人员：{r['contact']} ｜ 视频：{r['video']} ｜ 视频时间：{r['video_time']:.1f}s")
            st.write(f"连续确认耗时（视频时间）：{r['confirmation_video_seconds']:.1f}s；从分析开始至告警（实际耗时）：{r['analysis_elapsed_seconds']:.2f}s")
            if r.get('ack_at'):
                st.write(f"告警到接警：{(datetime.fromisoformat(r['ack_at'])-datetime.fromisoformat(r['created_at'])).total_seconds():.1f}s")
            evidence=store.root/r['evidence']
            if evidence.exists():st.image(str(evidence),use_container_width=True)
            st.markdown('**模拟短信预览**')
            st.info(r['sms']['body']);st.caption(r['sms']['status']+'；不提供真实送达时间。')
            if r['status']!='已结案':
                with st.form(r['id']):
                    operator=st.text_input('处置人员姓名')
                    verdict=st.selectbox('核查结论',['确认火情','误报','无法确认']) if r['status']=='处理中' else ''
                    note=st.text_area('处置说明（结案必填）')
                    action='接警' if r['status']=='待接警' else '结案'
                    if st.form_submit_button('确认'+action):
                        try:store.transition(r['id'],action,operator,note,verdict);st.rerun()
                        except ValueError as e:st.error(str(e))
            st.markdown('**处置时间线**')
            if not r['history']:st.caption('等待值班人员接警。')
            for entry in r['history']:
                st.text(f"{entry['at']}  ·  {entry['operator']}  ·  {entry['action']}")
                if entry['note']:st.text(entry['note'])

with image_tab:
    st.markdown('<div class="section-kicker">IMAGE INSPECTION</div>',unsafe_allow_html=True)
    st.subheader('现场图片快速检查')
    st.caption('上传车棚现场照片，查看模型标注的疑似火情位置。')
    picture=st.file_uploader('上传车棚图片',type=['jpg','jpeg','png'],key='picture')
    if st.button('检测图片',disabled=picture is None):
        try:
            import numpy as np
            from PIL import Image,ImageOps
            im=ImageOps.exif_transpose(Image.open(picture)).convert('RGB')
            if im.width*im.height>20000000:raise ValueError('图片超过2000万像素')
            score,marked=predict(np.asarray(im)[:,:,::-1].copy(),threshold)
            st.image(marked,channels='BGR',use_container_width=True)
            st.write('发现疑似火情' if score>=threshold else '本次未检出达到阈值的火情')
            st.caption('图片检测仅作辅助展示，不自动生成视频告警事件。')
        except Exception as e:st.error(str(e))
st.caption('现有模型为原型基础，未检出不代表现场安全；纯烟雾能力及车棚场景效果尚需验证。')
