from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, precision_score, recall_score, average_precision_score

THRESHOLD = 0.90
FEATURES = ['eye_gaze_angle','audio_db','tab_switches','gaze_rolling_10s','audio_rolling_10s']

st.set_page_config(page_title='Project Panopticon | EduGuard AI', page_icon='🛡️', layout='wide')
st.title('🛡️ Project Panopticon')
st.subheader('EduGuard AI – Intelligent Proctoring AI')
st.caption('Time-series proctoring pipeline with a class-balanced Random Forest and a 90% human-review threshold.')
st.info('This is a separate web interface for the original Project Panopticon notebook. The original notebook and CSV files in this folder are preserved unchanged.')

@st.cache_data
def load_csv(uploaded, fallback_path):
    return pd.read_csv(uploaded) if uploaded is not None else pd.read_csv(fallback_path)

def prepare_data(video_df, sys_df):
    video_df = video_df.copy(); sys_df = sys_df.copy()
    video_df['timestamp'] = pd.to_datetime(video_df['timestamp'])
    sys_df['timestamp'] = pd.to_datetime(sys_df['timestamp'])
    video_df = video_df.sort_values('timestamp').reset_index(drop=True)
    sys_df = sys_df.sort_values('timestamp').reset_index(drop=True)
    merged_df = pd.merge_asof(video_df, sys_df, on='timestamp', direction='backward', tolerance=pd.Timedelta('1s'))
    merged_df['eye_gaze_angle'] = merged_df['eye_gaze_angle'].ffill()
    merged_df['audio_db'] = merged_df['audio_db'].interpolate(method='linear', limit_direction='both')
    merged_df[['tab_switches','is_cheating']] = merged_df[['tab_switches','is_cheating']].fillna(0)
    merged_df['eye_gaze_angle'] = merged_df['eye_gaze_angle'].bfill()
    merged_df['gaze_rolling_10s'] = merged_df['eye_gaze_angle'].rolling(window=10).mean()
    merged_df['audio_rolling_10s'] = merged_df['audio_db'].rolling(window=10).max()
    merged_df = merged_df.dropna().reset_index(drop=True)
    merged_df['is_cheating'] = merged_df['is_cheating'].astype(int)
    return merged_df

def train_and_evaluate(df):
    X, y = df[FEATURES], df['is_cheating']
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    model = RandomForestClassifier(n_estimators=300, class_weight='balanced_subsample', min_samples_leaf=5, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:,1]
    pred_default = (probabilities >= 0.50).astype(int)
    pred_strict = (probabilities >= THRESHOLD).astype(int)
    return model, X_test, y_test, probabilities, pred_default, pred_strict

with st.sidebar:
    st.header('Data')
    video_upload = st.file_uploader('Video telemetry CSV', type=['csv'])
    events_upload = st.file_uploader('System events CSV', type=['csv'])
    st.divider(); st.metric('Decision threshold', f'{THRESHOLD:.0%}')
    st.caption('A flag is produced when model P(cheating) ≥ 90%.')

base = Path(__file__).resolve().parent
try:
    video_df = load_csv(video_upload, base/'video_telemetry.csv')
    sys_df = load_csv(events_upload, base/'system_events.csv')
    required_video = {'timestamp','eye_gaze_angle','audio_db'}
    required_events = {'timestamp','tab_switches','is_cheating'}
    if not required_video.issubset(video_df.columns):
        st.error(f'Video CSV must contain: {sorted(required_video)}'); st.stop()
    if not required_events.issubset(sys_df.columns):
        st.error(f'System events CSV must contain: {sorted(required_events)}'); st.stop()

    merged_df = prepare_data(video_df, sys_df)
    model, X_test, y_test, probabilities, pred_default, pred_strict = train_and_evaluate(merged_df)
    tn50, fp50, fn50, tp50 = confusion_matrix(y_test, pred_default, labels=[0,1]).ravel()
    tn90, fp90, fn90, tp90 = confusion_matrix(y_test, pred_strict, labels=[0,1]).ravel()
    prec50 = precision_score(y_test, pred_default, zero_division=0); rec50 = recall_score(y_test, pred_default, zero_division=0)
    prec90 = precision_score(y_test, pred_strict, zero_division=0); rec90 = recall_score(y_test, pred_strict, zero_division=0)

    t1,t2,t3,t4 = st.tabs(['📊 Dashboard','📈 Analytics','🧠 Model','📋 Data Preview'])
    with t1:
        st.success('Pipeline executed successfully.')
        c1,c2,c3,c4 = st.columns(4)
        c1.metric('Processed rows', f'{len(merged_df):,}'); c2.metric('Test rows', f'{len(y_test):,}')
        c3.metric('Cheating seconds in test', f'{int(y_test.sum()):,}'); c4.metric('Strict precision', f'{prec90:.1%}')
        st.subheader('Threshold comparison')
        st.dataframe(pd.DataFrame({'Metric':['False positives','False negatives','True positives','Precision','Recall'],'50% threshold':[fp50,fn50,tp50,prec50,rec50],'90% threshold':[fp90,fn90,tp90,prec90,rec90]}), use_container_width=True, hide_index=True)
        st.subheader('Current decision')
        st.write(f'The strict threshold is **{THRESHOLD:.0%}**. At that threshold, the test set produced **{fp90} false positives**, **{fn90} false negatives**, and **{tp90} true positives**.')
        st.warning('A model flag should be treated as a human-review signal, not as an automatic penalty.')

    with t2:
        precision, recall, thresholds = precision_recall_curve(y_test, probabilities)
        ap = average_precision_score(y_test, probabilities)
        fig, ax = plt.subplots(figsize=(10,5)); ax.plot(recall,precision,label=f'PR curve (AP = {ap:.3f})'); ax.scatter([rec90],[prec90],s=90,label=f'Operating point @ {THRESHOLD:.0%}'); ax.axhline(.90,linestyle=':',label='90% precision target'); ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); ax.set_title('Precision-Recall Curve'); ax.legend(loc='lower left'); st.pyplot(fig); plt.close(fig)
        fig2, ax2 = plt.subplots(figsize=(10,5)); ax2.plot(thresholds,precision[:-1],label='Precision'); ax2.plot(thresholds,recall[:-1],label='Recall'); ax2.axvline(THRESHOLD,linestyle='--',label=f'Strict threshold = {THRESHOLD:.0%}'); ax2.set_xlabel('Decision threshold'); ax2.set_ylabel('Score'); ax2.set_title('Precision & Recall vs Threshold'); ax2.legend(); st.pyplot(fig2); plt.close(fig2)

    with t3:
        st.subheader('Feature importance')
        st.bar_chart(pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=True))
        st.subheader('Classification report — 90% threshold')
        report = classification_report(y_test,pred_strict,target_names=['Innocent','Cheating'],output_dict=True,zero_division=0)
        st.dataframe(pd.DataFrame(report).transpose().round(3),use_container_width=True)
        st.subheader('Model configuration')
        st.code('RandomForestClassifier(\n    n_estimators=300,\n    class_weight="balanced_subsample",\n    min_samples_leaf=5,\n    random_state=42,\n    n_jobs=-1\n)', language='python')

    with t4:
        st.subheader('Merged and engineered data')
        st.dataframe(merged_df.head(100),use_container_width=True,hide_index=True)
        st.caption('Same pipeline: merge_asof with a 1-second backward tolerance, imputation, 10-second rolling gaze mean, and 10-second rolling audio maximum.')
except Exception as exc:
    st.error('The pipeline could not be executed.')
    st.exception(exc)
    st.info('If the error mentions a blocked sklearn/scipy .pyd or DLL, the Windows Application Control issue from the original local environment is still present. This web app does not bypass Windows security.')
