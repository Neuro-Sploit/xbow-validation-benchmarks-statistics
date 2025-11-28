import itertools
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
st.set_page_config(
    page_title='Benchmarks Analytics',
    layout='wide', page_icon='🛡️',
)
DATA_FILE = 'benchmarks.json'

OWASP_CATEGORIES = {
    'A01: Broken Access Control': ['access', 'admin', 'privilege', 'idor', 'bypass', 'authz', 'unauthorized'],
    'A02: Cryptographic Failures': ['crypto', 'encryption', 'hash', 'random', 'padding', 'cipher'],
    'A03: Injection': ['sql', 'sqli', 'injection', 'xss', 'command', 'rce', 'ssti', 'ldap', 'xpath'],
    'A04: Insecure Design': ['logic', 'business', 'design', 'flow'],
    'A05: Security Misconfiguration': ['config', 'default', 'misconfiguration', 'debug', 'header'],
    'A06: Vuln. Components': ['cve', 'version', 'outdated', 'dependency'],
    'A07: I & A Failures': ['auth', 'login', 'jwt', 'cookie', 'session', 'password', 'bruteforce'],
    'A08: Integrity Failures': ['deserialization', 'integrity', 'pickle', 'serialize'],
    'A09: Logging & Monitoring': ['log', 'monitor', 'audit'],
    'A10: SSRF': ['ssrf', 'request-forgery', 'metadata'],
}


@st.cache_data
def load_data():
    if not os.path.exists(DATA_FILE):
        return pd.DataFrame(), pd.DataFrame()

    df = pd.read_json(DATA_FILE)
    if df.empty:
        return df, df

    df = df.sort_values('id').reset_index(drop=True)
    df['tags_str'] = df['tags'].apply(
        lambda x: ', '.join(x) if isinstance(x, list) else '',
    )
    df['level_str'] = df['level'].astype(str)

    exploded = df.explode('tags')
    exploded['owasp_category'] = exploded['tags'].apply(map_tag_to_category)

    return df, exploded


def map_tag_to_category(tag):
    if not isinstance(tag, str):
        return 'Other'
    tag_lower = tag.lower()
    for category, keywords in OWASP_CATEGORIES.items():
        for keyword in keywords:
            if keyword in tag_lower:
                return category
    return 'Other'


def map_level_to_difficulty(level):
    """Map numeric level to difficulty label."""
    level_map = {1: 'easy', 2: 'medium', 3: 'hard'}
    return level_map.get(level, f'level_{level}')


def get_unique_tags(df):
    tags = set()
    for t in df['tags']:
        if isinstance(t, list):
            tags.update(t)
    return sorted(tags)


def render_metrics(df):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Total Benchmarks', len(df))
    c2.metric('Unique Levels', df['level'].nunique())
    c3.metric('Avg Tags/Problem', f"{df['tags'].apply(len).mean():.1f}")
    if 'points' in df.columns:
        c4.metric('Total Points', f"{df['points'].sum():,}")


def render_distributions(df, df_exploded):
    c1, c2 = st.columns(2)
    with c1:
        st.subheader('Difficulty Distribution')
        # Map numeric levels to difficulty labels
        df_with_difficulty = df.copy()
        df_with_difficulty['difficulty'] = df_with_difficulty['level'].apply(
            map_level_to_difficulty,
        )
        counts = df_with_difficulty['difficulty'].value_counts().reset_index()
        counts.columns = ['difficulty', 'count']
        # Order by difficulty: easy, medium, hard
        difficulty_order = ['easy', 'medium', 'hard']
        counts['difficulty'] = pd.Categorical(
            counts['difficulty'],
            categories=difficulty_order,
            ordered=True,
        )
        counts = counts.sort_values('difficulty')
        st.plotly_chart(
            px.pie(
                counts, values='count',
                names='difficulty', hole=0.4,
            ), width='stretch',
        )

    with c2:
        st.subheader('Top Tags')
        n = st.slider('Top N', 5, 50, 15)
        counts = df_exploded['tags'].value_counts().head(n).reset_index()
        counts.columns = ['tag', 'count']
        # Sort descending so most frequent tags appear at top
        counts = counts.sort_values('count', ascending=False)
        fig = px.bar(
            counts, x='count', y='tag',
            orientation='h',
        )
        # Reverse y-axis so highest count is at top
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, width='stretch')


def render_cooccurrence(df):
    st.subheader('Tag Co-occurrence Matrix')
    top_n = st.slider('Analyze Top N Tags', 5, 30, 20)

    all_tags = [t for tags in df['tags'] for t in tags]
    top_tags = pd.Series(all_tags).value_counts().head(top_n).index.tolist()

    matrix = pd.DataFrame(0, index=top_tags, columns=top_tags)
    for tags in df['tags']:
        relevant = [t for t in tags if t in top_tags]
        for t1, t2 in itertools.combinations(relevant, 2):
            matrix.loc[t1, t2] += 1
            matrix.loc[t2, t1] += 1

    fig = px.imshow(
        matrix,
        color_continuous_scale=[
            [0, '#f5f7fb'],  # light background for zero values
            [1, '#1f4b99'],
        ],
        aspect='equal',
        zmin=0,
    )
    # Keep cell size readable regardless of tag count
    cell_px = 48
    matrix_dim = max(320, len(top_tags) * cell_px)
    fig.update_layout(height=matrix_dim, width=matrix_dim)
    fig.update_traces(xgap=1, ygap=1)
    st.plotly_chart(fig, width='stretch')


def render_knowledge_map(df_exploded):
    st.subheader('OWASP Top 10 Coverage')
    counts = df_exploded[
        df_exploded['owasp_category'] !=
        'Other'
    ]['owasp_category'].value_counts().reset_index()
    counts.columns = ['category', 'count']

    c1, c2 = st.columns([1, 1])
    with c1:
        fig = go.Figure(
            data=go.Scatterpolar(
                r=counts['count'], theta=counts['category'], fill='toself',
            ),
        )
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True),
            ), showlegend=False,
        )
        st.plotly_chart(fig, width='stretch')
    with c2:
        st.dataframe(counts, width='stretch')


def render_problem_matrix(df):
    st.subheader('Problem-Tag Matrix')

    # Only show tags present in the filtered view to avoid empty columns
    current_tags = get_unique_tags(df)

    if len(df) > 300:
        st.warning(f"Displaying {len(df)} rows. Rendering might be slow.")

    # Prepare data for heatmap
    matrix_data = []
    ids = df['id'].tolist()

    for tags in df['tags']:
        row = [1 if t in tags else 0 for t in current_tags]
        matrix_data.append(row)

    fig = px.imshow(
        matrix_data,
        x=current_tags,
        y=ids,
        color_continuous_scale=[
            [0, '#f5f7fb'],  # Light base for missing data
            [1, '#636EFA'],
        ],
        aspect='equal',
        labels=dict(x='Tag', y='Benchmark', color='Present'),
        zmin=0,
        zmax=1,
    )
    fig.update_layout(
        height=max(480, len(ids) * 32),  # Larger cells for readability
        width=max(960, len(current_tags) * 32),
        xaxis={'side': 'top', 'tickangle': -45},
    )
    fig.update_traces(xgap=1, ygap=1)
    fig.update_traces(showscale=False)
    st.plotly_chart(fig, width='stretch')


def render_table(df):
    st.subheader('Data Table')
    cols = ['id', 'name', 'level', 'points', 'win_condition', 'tags_str']
    st.dataframe(df[cols], width='stretch')
    st.download_button(
        'Download CSV', df.to_csv(
            index=False,
        ), 'benchmarks.csv',
    )


def main():
    st.title('🛡️ Xbow Benchmark Analytics')

    df, df_exploded = load_data()
    if df.empty:
        st.error(f"Missing {DATA_FILE}")
        st.stop()

    # Sidebar
    st.sidebar.header('Filters')
    levels = st.sidebar.multiselect(
        'Level', sorted(
            df['level'].unique(),
        ), default=sorted(df['level'].unique()),
    )
    tags = st.sidebar.multiselect('Tags', get_unique_tags(df))

    # Filter Logic
    mask = df['level'].isin(levels)
    if tags:
        mask &= df['tags'].apply(lambda x: any(t in x for t in tags))

    filtered = df[mask]
    filtered_exploded = df_exploded[df_exploded['id'].isin(filtered['id'])]

    # Render
    render_metrics(filtered)
    st.divider()

    tabs = st.tabs([
        'Overview', 'Co-occurrence',
        'Knowledge Map', 'Problem Matrix', 'Data Table',
    ])

    with tabs[0]:
        render_distributions(filtered, filtered_exploded)
    with tabs[1]:
        render_cooccurrence(filtered)
    with tabs[2]:
        render_knowledge_map(filtered_exploded)
    with tabs[3]:
        render_problem_matrix(filtered)
    with tabs[4]:
        render_table(filtered)


if __name__ == '__main__':
    main()
