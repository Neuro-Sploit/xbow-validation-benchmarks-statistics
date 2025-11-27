import os

import pandas as pd
import plotly.express as px
import streamlit as st

# Set page config immediately
st.set_page_config(
    page_title='Benchmarks Analysis',
    layout='wide', page_icon='📊',
)

# --- Constants ---
DATA_FILE = 'benchmarks.json'

# --- Data Loading ---


@st.cache_data
def load_data():
    """
    Loads the pre-processed JSON file directly into a DataFrame.
    """
    if not os.path.exists(DATA_FILE):
        return pd.DataFrame()

    # Read JSON directly. Since we exported using model_dump(mode='json'),
    # 'tags' will be actual lists, and 'win_condition' will be strings.
    df = pd.read_json(DATA_FILE)

    if df.empty:
        return df

    # Ensure sorting
    df = df.sort_values('id').reset_index(drop=True)

    # Create a string version of tags for display in the table
    df['tags_str'] = df['tags'].apply(
        lambda x: ', '.join(x) if isinstance(x, list) else '',
    )

    # Create categorical string for levels (for charts)
    df['level_str'] = df['level'].astype(str)

    return df


def get_unique_tags(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return []
    tag_set = set()
    for tags in df['tags']:
        if isinstance(tags, list):
            tag_set.update(tags)
    return sorted(tag_set)

# --- Main Application ---


def main():
    st.title('📊 Benchmarks Analytics')
    st.markdown('Explore and visualize benchmark datasets.')

    df_full = load_data()

    if df_full.empty:
        st.error(
            f"Data file `{DATA_FILE}` not found. Please ensure it is committed to the repository.",
        )
        st.stop()

    # --- Sidebar Filters ---
    st.sidebar.header('🔍 Filters')

    # 1. Search
    search_query = st.sidebar.text_input(
        'Search (Name/Desc)', value='', placeholder='e.g. sql injection',
    )

    # 2. Level Filter
    available_levels = sorted(df_full['level'].unique().tolist())
    selected_levels = st.sidebar.multiselect(
        'Level', options=available_levels, default=available_levels,
    )

    # 3. Win Condition Filter
    # Handle explicit check if column exists
    if 'win_condition' in df_full.columns:
        available_wins = sorted(
            df_full['win_condition'].astype(str).unique().tolist(),
        )
        selected_wins = st.sidebar.multiselect(
            'Win Condition', options=available_wins, default=available_wins,
        )
    else:
        selected_wins = []

    # 4. Tag Filter
    all_tags = get_unique_tags(df_full)
    selected_tags = st.sidebar.multiselect(
        'Tags', options=all_tags, default=[],
    )
    require_all_tags = st.sidebar.checkbox(
        'Match all selected tags', value=False,
    )

    # --- Apply Filters ---
    filtered = df_full.copy()

    # Filter: Level
    filtered = filtered[filtered['level'].isin(selected_levels)]

    # Filter: Win Condition
    if 'win_condition' in filtered.columns:
        filtered = filtered[
            filtered['win_condition'].astype(
                str,
            ).isin(selected_wins)
        ]

    # Filter: Tags
    if selected_tags:
        if require_all_tags:
            filtered = filtered[
                filtered['tags'].apply(
                    lambda ts: all(t in ts for t in selected_tags),
                )
            ]
        else:
            filtered = filtered[
                filtered['tags'].apply(
                    lambda ts: any(t in ts for t in selected_tags),
                )
            ]

    # Filter: Text Search
    if search_query.strip():
        q = search_query.strip().lower()
        filtered = filtered[
            filtered['name'].str.lower().str.contains(q, na=False) |
            filtered['description'].str.lower().str.contains(q, na=False)
        ]

    # --- Metrics ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric('Total Benchmarks', len(filtered))
    m2.metric('Unique Levels', filtered['level'].nunique())

    total_points = filtered['points'].sum(
    ) if 'points' in filtered.columns else 0
    m3.metric('Total Points', f"{total_points:,}")

    active_tags_count = len(get_unique_tags(filtered))
    m4.metric('Active Tags', active_tags_count)

    # --- Tabs Layout ---
    tab_data, tab_charts, tab_matrix = st.tabs(
        ['📋 Data Browser', '📈 Visualizations', '🧩 Problem Matrix'],
    )

    # TAB 1: Data Browser
    with tab_data:
        st.subheader('Filtered Dataset')

        # Columns to display
        show_cols = [
            'id', 'name', 'level', 'points',
            'win_condition', 'tags_str', 'description',
        ]
        # Filter strictly existing columns
        show_cols = [c for c in show_cols if c in filtered.columns]

        st.dataframe(
            filtered[show_cols],
            width='stretch',
            hide_index=True,
            column_config={
                'name': st.column_config.TextColumn('Name', width='medium'),
                'description': st.column_config.TextColumn('Description', width='large'),
                'tags_str': st.column_config.TextColumn('Tags'),
            },
        )

        # CSV Download
        st.download_button(
            label='Download Filtered Data (CSV)',
            data=filtered.drop(
                columns=['tags', 'level_str'], errors='ignore',
            ).to_csv(index=False),
            file_name='benchmarks_filtered.csv',
            mime='text/csv',
        )

    # TAB 2: Visualizations
    with tab_charts:
        c1, c2 = st.columns(2)

        with c1:
            st.caption('Distribution by Level')
            level_counts = filtered['level_str'].value_counts().reset_index()
            level_counts.columns = ['level', 'count']
            if not level_counts.empty:
                fig_pie = px.pie(
                    level_counts, values='count',
                    names='level', hole=0.4,
                )
                st.plotly_chart(fig_pie, width='stretch')
            else:
                st.info('No data available.')

        with c2:
            st.caption('Top Tags (Filtered)')
            if not filtered.empty:
                exploded = filtered.explode('tags')
                tag_counts = exploded['tags'].value_counts().reset_index()
                tag_counts.columns = ['tag', 'count']

                # Slider for top N
                top_n = st.slider('Number of tags to show', 5, 30, 10)
                fig_bar = px.bar(
                    tag_counts.head(top_n), x='count',
                    y='tag', orientation='h', text_auto=True,
                )
                fig_bar.update_layout(
                    yaxis={'categoryorder': 'total ascending'},
                )
                st.plotly_chart(fig_bar, width='stretch')
            else:
                st.info('No tags available.')

    # TAB 3: Matrix
    with tab_matrix:
        st.markdown('### Problem-Tag Matrix')
        st.caption('Heatmap showing which tags are present in which benchmark.')

        if not filtered.empty and all_tags:
            # Only use tags present in the current filter to keep matrix clean
            current_tags = get_unique_tags(filtered)

            # Limit warning
            if len(filtered) > 300:
                st.warning(
                    f"Displaying matrix for {len(filtered)} rows. This might be slow.",
                )

            matrix_data = []
            ids = []

            for _, row in filtered.iterrows():
                ids.append(row['id'])
                row_tags = set(row['tags']) if isinstance(
                    row['tags'], list,
                ) else set()
                matrix_data.append(
                    [1 if t in row_tags else 0 for t in current_tags],
                )

            # Use Plotly
            fig_hm = px.imshow(
                matrix_data,
                x=current_tags,
                y=ids,
                color_continuous_scale=[[0, 'white'], [1, '#4A90E2']],
                aspect='auto',
            )
            fig_hm.update_layout(
                height=max(500, len(ids) * 15),  # Dynamic height
                xaxis={'side': 'top', 'tickangle': -45},
                yaxis={'title': 'Benchmark ID'},
            )
            fig_hm.update_traces(showscale=False)
            st.plotly_chart(fig_hm, width='stretch')


if __name__ == '__main__':
    main()
