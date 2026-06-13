import os
import io
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import httpx

API_BASE = os.getenv("API_BASE", "http://localhost:8000/api/v1")

st.set_page_config(page_title="Text Augment Platform", page_icon="🧪", layout="wide")


def api_get(path: str, params: dict = None):
    try:
        r = httpx.get(f"{API_BASE}{path}", params=params, timeout=30)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def api_post(path: str, data: dict = None, files=None, timeout=30):
    try:
        if files:
            r = httpx.post(f"{API_BASE}{path}", data=data, files=files, timeout=timeout)
        else:
            r = httpx.post(f"{API_BASE}{path}", json=data, timeout=timeout)
        return r.json() if r.status_code in (200, 201) else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


def api_delete(path: str):
    try:
        r = httpx.delete(f"{API_BASE}{path}", timeout=30)
        return r.status_code == 200
    except Exception:
        return False


st.sidebar.title("🧪 Text Augment Platform")
page = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard", "📁 Datasets", "🔧 Augmentation", "🔍 Filtering", "🏋️ Training", "📈 Evaluation"],
)

if page == "📊 Dashboard":
    st.title("📊 Experiment Dashboard")

    datasets = api_get("/datasets") or []
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Datasets", len(datasets))
    total_samples = sum(d.get("total_samples", 0) for d in datasets)
    col2.metric("Total Samples", total_samples)

    aug_tasks = api_get("/augmentation/tasks") or []
    running_aug = sum(1 for t in aug_tasks if t.get("status") == "running")
    completed_aug = sum(1 for t in aug_tasks if t.get("status") == "completed")
    col3.metric("Aug Tasks (Running)", running_aug)
    col4.metric("Aug Tasks (Done)", completed_aug)

    exps = api_get("/training/experiments") or []
    running_exp = sum(1 for e in exps if e.get("status") == "running")
    completed_exp = sum(1 for e in exps if e.get("status") == "completed")
    col1.metric("Training Experiments", len(exps))
    col2.metric("Running Experiments", running_exp)
    col3.metric("Completed Experiments", completed_exp)

    st.subheader("Recent Datasets")
    if datasets:
        df = pd.DataFrame(datasets)
        display_cols = ["id", "name", "total_samples", "num_classes", "min_class_samples", "imbalance_ratio", "version_count"]
        available = [c for c in display_cols if c in df.columns]
        st.dataframe(df[available], use_container_width=True)
    else:
        st.info("No datasets yet. Upload one in the Datasets tab.")

    st.subheader("Recent Experiments")
    if exps:
        exp_df = pd.DataFrame(exps)
        disp = ["id", "experiment_name", "training_mode", "backbone", "status", "best_val_metric"]
        avail = [c for c in disp if c in exp_df.columns]
        st.dataframe(exp_df[avail], use_container_width=True)
    else:
        st.info("No experiments yet.")

elif page == "📁 Datasets":
    st.title("📁 Dataset Management")

    tab_upload, tab_unlabeled, tab_list, tab_detail = st.tabs(["Upload", "Import Unlabeled", "List", "Detail"])

    with tab_upload:
        st.subheader("Upload Dataset")
        name = st.text_input("Dataset Name", key="upload_name")
        description = st.text_area("Description", key="upload_desc")
        file_format = st.selectbox("File Format", ["csv", "json"])
        text_col = st.text_input("Text Column", "text")
        label_col = st.text_input("Label Column", "label")

        col1, col2, col3 = st.columns(3)
        with col1:
            train_r = st.number_input("Train Ratio", 0.1, 0.9, 0.7, 0.05)
        with col2:
            val_r = st.number_input("Val Ratio", 0.05, 0.5, 0.15, 0.05)
        with col3:
            test_r = st.number_input("Test Ratio", 0.05, 0.5, 0.15, 0.05)

        uploaded_file = st.file_uploader("Upload File", type=["csv", "json"])
        if st.button("Import Dataset") and uploaded_file and name:
            with st.spinner("Importing..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/octet-stream")}
                data = {
                    "name": name,
                    "description": description,
                    "text_column": text_col,
                    "label_column": label_col,
                    "train_ratio": str(train_r),
                    "val_ratio": str(val_r),
                    "test_ratio": str(test_r),
                }
                result = api_post("/datasets/import", data=data, files=files)
                if "error" in result:
                    st.error(f"Import failed: {result['error']}")
                else:
                    st.success(f"Dataset imported! ID: {result.get('dataset_id')}, Samples: {result.get('total_samples')}")
                    st.json(result)

    with tab_unlabeled:
        st.subheader("Import Unlabeled Data (for Semi-supervised Training)")
        datasets = api_get("/datasets") or []
        if not datasets:
            st.warning("No datasets available. Please upload a labeled dataset first.")
        else:
            ds_opts = {f"{d['id']}: {d['name']}": d["id"] for d in datasets}
            sel_ds = st.selectbox("Select Dataset", list(ds_opts.keys()), key="unlabeled_ds")
            ds_id = ds_opts[sel_ds]

            ul_file_format = st.selectbox("File Format", ["csv", "json", "txt"], key="ul_format")
            ul_text_col = st.text_input("Text Column", "text", key="ul_text_col")

            ul_file = st.file_uploader("Upload Unlabeled File", type=["csv", "json", "txt"], key="ul_file")
            if st.button("Import Unlabeled Data") and ul_file:
                with st.spinner("Importing unlabeled data..."):
                    files = {"file": (ul_file.name, ul_file.getvalue(), "application/octet-stream")}
                    data = {
                        "text_column": ul_text_col,
                    }
                    result = api_post(f"/datasets/{ds_id}/import-unlabeled", data=data, files=files)
                    if "error" in result:
                        st.error(f"Import failed: {result['error']}")
                    else:
                        st.success(f"Unlabeled data imported! Version ID: {result.get('version_id')}, Samples: {result.get('total_samples')}")
                        st.json(result)

    with tab_list:
        st.subheader("All Datasets")
        datasets = api_get("/datasets") or []
        if datasets:
            df = pd.DataFrame(datasets)
            display_cols = ["id", "name", "description", "total_samples", "num_classes", "min_class_samples", "imbalance_ratio", "version_count"]
            available = [c for c in display_cols if c in df.columns]
            st.dataframe(df[available], use_container_width=True)
        else:
            st.info("No datasets found.")

    with tab_detail:
        datasets = api_get("/datasets") or []
        if datasets:
            ds_options = {f"{d['id']}: {d['name']}": d["id"] for d in datasets}
            selected = st.selectbox("Select Dataset", list(ds_options.keys()))
            if selected:
                ds_id = ds_options[selected]
                detail = api_get(f"/datasets/{ds_id}")
                if detail:
                    st.json(detail)

                    versions = detail.get("versions", [])
                    if versions:
                        st.subheader("Versions")
                        v_df = pd.DataFrame(versions)
                        st.dataframe(v_df, use_container_width=True)

                    if st.button(f"Delete Dataset {ds_id}"):
                        if api_delete(f"/datasets/{ds_id}"):
                            st.success("Deleted!")
                        else:
                            st.error("Delete failed.")

                    st.subheader("Version Samples")
                    if versions:
                        v_opts = {f"v{v['id']}: {v['version_name']}": v["id"] for v in versions}
                        sel_v = st.selectbox("Select Version", list(v_opts.keys()), key="sample_version")
                        if sel_v:
                            v_id = v_opts[sel_v]
                            split_filter = st.selectbox("Split", ["All", "train", "val", "test"], key="sample_split")
                            source_filter = st.selectbox("Source", ["All", "original", "synonym_replacement", "random_ops", "back_translation", "context_augment", "template_generation"], key="sample_source")
                            params = {}
                            if split_filter != "All":
                                params["split"] = split_filter
                            if source_filter != "All":
                                params["source"] = source_filter

                            samples = api_get(f"/datasets/versions/{v_id}/samples", params=params)
                            if samples and "samples" in samples:
                                st.info(f"Total: {samples['total']} samples")
                                s_df = pd.DataFrame(samples["samples"])
                                st.dataframe(s_df, use_container_width=True)
                            else:
                                st.info("No samples found.")

                    if len(versions) >= 2:
                        st.subheader("Compare Versions")
                        v_ids = [v["id"] for v in versions]
                        v_names = [f"v{v['id']}: {v['version_name']}" for v in versions]
                        c1, c2 = st.columns(2)
                        with c1:
                            va_name = st.selectbox("Version A", v_names, key="cmp_a")
                            va_idx = v_names.index(va_name)
                        with c2:
                            vb_name = st.selectbox("Version B", v_names, index=min(1, len(v_names)-1), key="cmp_b")
                            vb_idx = v_names.index(vb_name)

                        if st.button("Compare"):
                            va_id = v_ids[va_idx]
                            vb_id = v_ids[vb_idx]
                            cmp = api_get(f"/datasets/versions/compare", params={"version_id_a": va_id, "version_id_b": vb_id})
                            if cmp:
                                st.write(f"Sample count diff: {cmp.get('sample_count_diff', 'N/A')}")
                                dist_diff = cmp.get("distribution_diff", {})
                                if dist_diff:
                                    dd_df = pd.DataFrame(dist_diff).T
                                    st.dataframe(dd_df, use_container_width=True)

elif page == "🔧 Augmentation":
    st.title("🔧 Augmentation Strategy Engine")

    tab_create, tab_tasks = st.tabs(["Create Task", "Task Monitor"])

    with tab_create:
        datasets = api_get("/datasets") or []
        if not datasets:
            st.warning("No datasets available. Please upload one first.")
        else:
            ds_opts = {f"{d['id']}: {d['name']}": d["id"] for d in datasets}
            sel_ds = st.selectbox("Dataset", list(ds_opts.keys()))
            ds_id = ds_opts[sel_ds]

            detail = api_get(f"/datasets/{ds_id}")
            versions = detail.get("versions", []) if detail else []
            v_opts = {f"v{v['id']}: {v['version_name']} ({v.get('total_samples',0)} samples)": v["id"] for v in versions}
            sel_v = st.selectbox("Source Version", list(v_opts.keys()))
            source_v_id = v_opts[sel_v]

            mode = st.radio("Mode", ["Single Strategy", "Composite Pipeline"], horizontal=True)
            multiplier = st.number_input("Augmentation Multiplier", 0.1, 10.0, 1.0, 0.5)

            def _get_strategy_params(strategy_name, key_prefix=""):
                params = {}
                if strategy_name == "synonym_replacement":
                    params["replace_ratio"] = st.slider(f"Replace Ratio", 0.01, 0.5, 0.1, 0.01, key=f"{key_prefix}replace_ratio")
                    params["language"] = st.selectbox(f"Language", ["en", "zh"], key=f"{key_prefix}lang")
                elif strategy_name == "random_ops":
                    params["n_ops"] = st.number_input(f"Number of Operations (0=auto)", 0, 20, 0, key=f"{key_prefix}n_ops")
                    if params["n_ops"] == 0:
                        params["n_ops"] = None
                    params["delete_prob"] = st.slider(f"Delete Probability", 0.05, 0.5, 0.1, 0.05, key=f"{key_prefix}del_prob")
                    params["language"] = st.selectbox(f"Language", ["en"], key=f"{key_prefix}rand_lang")
                elif strategy_name == "back_translation":
                    params["source_language"] = st.selectbox(f"Source Language", ["en", "zh", "de", "fr", "ja"], key=f"{key_prefix}bt_src")
                    use_custom_pivot = st.checkbox(f"Use custom pivot language (advanced)", value=False, key=f"{key_prefix}custom_pivot")
                    if use_custom_pivot:
                        params["pivot_language"] = st.text_input(
                            f"Pivot Language Code",
                            value="fr",
                            key=f"{key_prefix}pivot_code",
                            help="e.g., 'fr' for French, 'es' for Spanish, 'de' for German. "
                                 "Note: Pivot must be from a different language family than source."
                        )
                    else:
                        pivot_options = {
                            "en": ["fr", "de", "es", "zh", "ja"],
                            "zh": ["en", "fr", "de", "ja"],
                            "de": ["fr", "es", "zh", "ja"],
                            "fr": ["en", "de", "zh", "ja"],
                            "ja": ["en", "zh", "fr", "de"],
                        }
                        pivots = pivot_options.get(params["source_language"], ["en", "fr"])
                        params["pivot_language"] = st.selectbox(f"Pivot Language", pivots, key=f"{key_prefix}pivot")
                    params["num_variants"] = st.number_input(f"Number of Variants", 1, 5, 1, key=f"{key_prefix}num_variants")
                elif strategy_name == "context_augment":
                    params["mask_ratio"] = st.slider(f"Mask Ratio", 0.05, 0.5, 0.15, 0.05, key=f"{key_prefix}mask_ratio")
                    params["top_k"] = st.number_input(f"Top-K Sampling", 1, 20, 5, key=f"{key_prefix}top_k")
                    params["num_variants"] = st.number_input(f"Number of Variants", 1, 10, 1, key=f"{key_prefix}num_vars")
                    params["model_name"] = st.selectbox(f"MLM Model", ["bert-base-uncased", "bert-base-chinese"], key=f"{key_prefix}model")
                elif strategy_name == "template_generation":
                    params["template"] = st.text_input(f"Template", value="{label}类的例子: {text}", key=f"{key_prefix}template")
                    params["samples_per_seed"] = st.number_input(f"Samples Per Seed", 1, 20, 3, key=f"{key_prefix}samples_per_seed")
                return params

            is_composite = mode == "Composite Pipeline"
            steps = []

            if is_composite:
                st.subheader("Pipeline Steps")
                st.info("💡 Constraints: Back-translation must be first step. "
                        "Context augmentation cannot follow random_ops immediately.")
                num_steps = st.number_input("Number of Steps", 1, 10, 2, 1)
                strategy_list = ["synonym_replacement", "random_ops", "back_translation", "context_augment", "template_generation"]

                for i in range(int(num_steps)):
                    with st.expander(f"Step {i + 1}", expanded=True):
                        step_strategy = st.selectbox(
                            f"Strategy (Step {i+1})",
                            strategy_list,
                            key=f"step_strat_{i}",
                        )
                        step_params = _get_strategy_params(step_strategy, key_prefix=f"step_{i}_")
                        steps.append({"strategy": step_strategy, "strategy_params": step_params})
            else:
                strategy = st.selectbox(
                    "Augmentation Strategy",
                    ["synonym_replacement", "random_ops", "back_translation", "context_augment", "template_generation"],
                )
                st.subheader("Strategy Parameters")
                params = _get_strategy_params(strategy)

            col1, col2 = st.columns(2)
            with col1:
                preview_clicked = st.button("🔍 Preview Effect", type="secondary")
            with col2:
                create_clicked = st.button("✨ Create Augmentation Task", type="primary")

            if preview_clicked:
                with st.spinner("Generating preview..."):
                    if is_composite:
                        st.info("Preview for composite pipeline shows the first step result only.")
                        first_step = steps[0] if steps else None
                        if first_step:
                            preview_strategy = first_step["strategy"]
                            preview_params = first_step["strategy_params"]
                        else:
                            st.warning("Please add at least one step.")
                            st.stop()
                    else:
                        preview_strategy = strategy
                        preview_params = params

                preview_payload = {
                    "source_version_id": source_v_id,
                    "strategy": preview_strategy,
                    "strategy_params": preview_params,
                }
                preview_result = api_post("/augmentation/preview", data=preview_payload, timeout=15)
                if preview_result and "error" not in preview_result:
                    st.subheader(f"Preview Results - {preview_result.get('strategy')}")
                    st.caption(f"Success: {preview_result.get('success_count', 0)}/{preview_result.get('total_count', 0)}, "
                               f"Timed out: {preview_result.get('timed_out_count', 0)}")

                    samples = preview_result.get("samples", [])
                    for idx, s in enumerate(samples):
                        with st.container():
                            st.markdown(f"**Sample {idx + 1}**")
                            c1, c2 = st.columns(2)
                            with c1:
                                st.info("Original")
                                st.write(s.get("original_text", ""))
                            with c2:
                                if s.get("timed_out"):
                                    st.warning("⏱️ Timed out")
                                elif s.get("error"):
                                    st.error(f"Error: {s.get('error')}")
                                elif s.get("augmented_text"):
                                    st.success("Augmented")
                                    st.write(s.get("augmented_text"))
                                else:
                                    st.info("No change")
                                    st.write(s.get("original_text", ""))
                            st.divider()
                else:
                    error_msg = preview_result.get("error", "Unknown error") if preview_result else "No response"
                    st.error(f"Preview failed: {error_msg}")

            if create_clicked:
                if is_composite:
                    step_list = [{"strategy": s["strategy"], "strategy_params": s["strategy_params"]} for s in steps]
                    payload = {
                        "dataset_id": ds_id,
                        "source_version_id": source_v_id,
                        "strategy": "composite",
                        "strategy_params": {},
                        "augmentation_multiplier": multiplier,
                        "is_composite": True,
                        "steps": step_list,
                    }
                else:
                    payload = {
                        "dataset_id": ds_id,
                        "source_version_id": source_v_id,
                        "strategy": strategy,
                        "strategy_params": params,
                        "augmentation_multiplier": multiplier,
                        "is_composite": False,
                        "steps": [],
                    }
                result = api_post("/augmentation/tasks", data=payload)
                if "error" in result:
                    st.error(f"Failed: {result['error']}")
                else:
                    st.success(f"Task created! ID: {result.get('id')}")
                    st.json(result)

    with tab_tasks:
        st.subheader("Augmentation Tasks")
        tasks = api_get("/augmentation/tasks") or []
        if tasks:
            t_df = pd.DataFrame(tasks)
            disp_cols = ["id", "dataset_id", "strategy", "status", "total_samples", "processed_samples", "generated_samples", "estimated_remaining_seconds"]
            avail = [c for c in disp_cols if c in t_df.columns]
            st.dataframe(t_df[avail], use_container_width=True)

            st.subheader("Task Detail & Actions")
            task_opts = {f"Task {t['id']}: {t['strategy']} ({t['status']})": t["id"] for t in tasks}
            sel_task = st.selectbox("Select Task", list(task_opts.keys()))
            if sel_task:
                tid = task_opts[sel_task]
                task_detail = api_get(f"/augmentation/tasks/{tid}")

                if task_detail:
                    if task_detail.get("is_composite") and task_detail.get("step_stats"):
                        st.subheader("Step Statistics")
                        step_stats = task_detail.get("step_stats", [])
                        if step_stats:
                            for s in step_stats:
                                with st.expander(f"Step {s.get('step_order', 0) + 1}: {s.get('strategy', '')}", expanded=True):
                                    c1, c2, c3 = st.columns(3)
                                    c1.metric("Input", s.get("input_count", 0))
                                    c2.metric("Success", s.get("success_count", 0))
                                    c3.metric("Skipped", s.get("skipped_count", 0))

                    if task_detail.get("is_composite") and task_detail.get("current_step_index") is not None:
                        st.info(f"Current step: Step {task_detail.get('current_step_index', 0) + 1}")

                col1, col2 = st.columns(2)
                with col1:
                    action = st.selectbox("Action", ["pause", "resume", "cancel"])
                with col2:
                    if st.button("Execute Action", type="primary"):
                        result = api_post(f"/augmentation/tasks/{tid}/action", data={"action": action})
                        st.json(result)
        else:
            st.info("No augmentation tasks yet.")

elif page == "🔍 Filtering":
    st.title("🔍 Quality Filtering")

    tab_create, tab_tasks = st.tabs(["Create Filter Task", "Task Results"])

    with tab_create:
        datasets = api_get("/datasets") or []
        if not datasets:
            st.warning("No datasets available.")
        else:
            all_versions = []
            for d in datasets:
                detail = api_get(f"/datasets/{d['id']}")
                if detail:
                    for v in detail.get("versions", []):
                        all_versions.append({
                            "id": v["id"],
                            "name": f"v{v['id']}: {d['name']} / {v['version_name']} ({v.get('total_samples',0)} samples)",
                        })

            if all_versions:
                v_opts = {v["name"]: v["id"] for v in all_versions}
                sel_v = st.selectbox("Version to Filter", list(v_opts.keys()))
                v_id = v_opts[sel_v]

                strictness = st.selectbox("Filter Strictness", ["standard", "loose", "strict"])

                st.subheader("Filter Presets")
                presets = api_get("/filtering/presets") or {}
                if presets:
                    preset_df = pd.DataFrame(presets).T
                    preset_df.index.name = "Strictness"
                    st.dataframe(preset_df, use_container_width=True)

                if st.button("Create Filter Task"):
                    result = api_post("/filtering/tasks", data={"version_id": v_id, "strictness": strictness})
                    if "error" in result:
                        st.error(f"Failed: {result['error']}")
                    else:
                        st.success(f"Filter task created! ID: {result.get('id')}")
            else:
                st.warning("No versions available.")

    with tab_tasks:
        st.subheader("Filter Tasks")
        tasks = api_get("/filtering/tasks") or []
        if tasks:
            for t in tasks:
                with st.expander(f"Filter Task {t['id']} - {t['strictness']} - {t['status']}"):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total", t.get("total_samples", 0))
                    col2.metric("Passed", t.get("passed_samples", 0))
                    col3.metric("Filtered", t.get("filtered_samples", 0))

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("PPL Filtered", t.get("ppl_filtered", 0))
                    c2.metric("Label Filtered", t.get("label_filtered", 0))
                    c3.metric("Similarity Filtered", t.get("similarity_filtered", 0))
                    c4.metric("Dedup Filtered", t.get("dedup_filtered", 0))
        else:
            st.info("No filter tasks yet.")

elif page == "🏋️ Training":
    st.title("🏋️ Training Pipeline")

    tab_create, tab_exps = st.tabs(["Create Experiment", "Experiments"])

    with tab_create:
        datasets = api_get("/datasets") or []
        if not datasets:
            st.warning("No datasets available.")
        else:
            all_versions = []
            ds_map = {}
            for d in datasets:
                detail = api_get(f"/datasets/{d['id']}")
                ds_map[d['id']] = d['name']
                if detail:
                    for v in detail.get("versions", []):
                        if v.get("version_type") != "unlabeled":
                            vtype = v.get("version_type", "original")
                            status_icon = "✅" if vtype == "filtered" else "⚠️" if vtype == "augmented" else "⚠️"
                            all_versions.append({
                                "id": v["id"],
                                "dataset_id": d["id"],
                                "version_type": vtype,
                                "name": f"{status_icon} v{v['id']}: {d['name']} / {v['version_name']} [{vtype}] ({v.get('total_samples',0)} samples)",
                            })

            exp_name = st.text_input("Experiment Name", value="experiment_1")

            if all_versions:
                filtered_only = [v for v in all_versions if v["version_type"] == "filtered"]
                non_filtered = [v for v in all_versions if v["version_type"] != "filtered"]

                if filtered_only:
                    st.success(f"✅ {len(filtered_only)} filtered version(s) available for training")
                if non_filtered:
                    st.warning(f"⚠️ {len(non_filtered)} version(s) not yet filtered - cannot be used for training")

                v_opts = {v["name"]: v["id"] for v in all_versions}
                sel_v = st.selectbox("Dataset Version", list(v_opts.keys()))
                v_id = v_opts[sel_v]
                sel_version = next(v for v in all_versions if v["id"] == v_id)
                sel_ds_id = next(v["dataset_id"] for v in all_versions if v["id"] == v_id)

                if sel_version["version_type"] != "filtered":
                    st.error(f"❌ Selected version is of type '{sel_version['version_type']}'. "
                             "Only 'filtered' versions can be used for training. "
                             "Please run a quality filter task on this version first.")

                training_mode = st.selectbox(
                    "Training Mode",
                    ["baseline", "augmented", "curriculum", "semi_supervised"],
                )
                backbone = st.selectbox(
                    "Model Backbone",
                    ["distilbert", "tinybert", "textcnn", "bilstm_attention"],
                )

                unlabeled_version_id = None
                if training_mode == "semi_supervised":
                    st.info("📋 Semi-supervised training requires an unlabeled data version.")
                    unlabeled_versions = []
                    for d in datasets:
                        detail = api_get(f"/datasets/{d['id']}")
                        if detail:
                            for v in detail.get("versions", []):
                                if v.get("version_type") == "unlabeled":
                                    unlabeled_versions.append({
                                        "id": v["id"],
                                        "name": f"v{v['id']}: {d['name']} / {v['version_name']} ({v.get('total_samples',0)} samples)",
                                    })

                    if unlabeled_versions:
                        ul_opts = {v["name"]: v["id"] for v in unlabeled_versions}
                        sel_ul = st.selectbox("Unlabeled Version", list(ul_opts.keys()), key="sel_ul")
                        unlabeled_version_id = ul_opts[sel_ul]
                    else:
                        st.warning("⚠️ No unlabeled versions available. Please import unlabeled data first in the Datasets tab.")

                st.subheader("Hyperparameters")
                lr = st.number_input("Learning Rate", 1e-7, 1e-2, 2e-5, format="%e")
                batch_size = st.number_input("Batch Size", 4, 128, 16)
                epochs = st.number_input("Epochs", 1, 100, 10)
                patience = st.number_input("Early Stopping Patience", 1, 20, 3)
                max_seq = st.number_input("Max Seq Length", 32, 512, 128)

                aug_multiplier = 1.0
                if training_mode == "augmented":
                    aug_multiplier = st.selectbox("Augmentation Multiplier", [0.5, 1.0, 2.0, 3.0], index=1)

                if st.button("Start Training"):
                    if training_mode == "semi_supervised" and unlabeled_version_id is None:
                        st.error("Please select an unlabeled version for semi-supervised training.")
                    else:
                        payload = {
                            "experiment_name": exp_name,
                            "dataset_id": sel_ds_id,
                            "version_id": v_id,
                            "unlabeled_version_id": unlabeled_version_id,
                            "training_mode": training_mode,
                            "backbone": backbone,
                            "hyperparams": {
                                "learning_rate": lr,
                                "batch_size": batch_size,
                                "epochs": epochs,
                                "early_stopping_patience": patience,
                                "max_seq_length": max_seq,
                            },
                            "augmentation_multiplier": aug_multiplier,
                        }
                        result = api_post("/training/experiments", data=payload)
                        if "error" in result:
                            st.error(f"Failed: {result['error']}")
                        else:
                            st.success(f"Experiment created! ID: {result.get('id')}")
            else:
                st.warning("No versions available.")

    with tab_exps:
        st.subheader("Training Experiments")
        exps = api_get("/training/experiments") or []
        if exps:
            for exp in exps:
                with st.expander(f"#{exp['id']} {exp['experiment_name']} - {exp['training_mode']} - {exp['status']}"):
                    c1, c2, c3 = st.columns(3)
                    c1.write(f"**Backbone:** {exp.get('backbone')}")
                    c2.write(f"**Epoch:** {exp.get('current_epoch', 0)}/{exp.get('total_epochs', '?')}")
                    c3.write(f"**Best Val Metric:** {exp.get('best_val_metric', 'N/A')}")

                    train_losses = exp.get("train_loss_history", [])
                    val_losses = exp.get("val_loss_history", [])
                    val_metrics = exp.get("val_metric_history", [])

                    if train_losses:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(y=train_losses, name="Train Loss", mode="lines"))
                        fig.add_trace(go.Scatter(y=val_losses, name="Val Loss", mode="lines"))
                        fig.update_layout(title="Loss Curves", xaxis_title="Epoch", yaxis_title="Loss")
                        st.plotly_chart(fig, use_container_width=True)

                    if val_metrics:
                        fig2 = go.Figure()
                        fig2.add_trace(go.Scatter(y=val_metrics, name="Val Metric", mode="lines"))
                        fig2.update_layout(title="Validation Metric", xaxis_title="Epoch", yaxis_title="Metric")
                        st.plotly_chart(fig2, use_container_width=True)

                    if exp.get("status") == "completed":
                        eval_result = api_get(f"/training/experiments/{exp['id']}/evaluation")
                        if eval_result and "error" not in eval_result:
                            st.subheader("Evaluation Results")
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Accuracy", f"{eval_result.get('accuracy', 0):.4f}")
                            c2.metric("Macro F1", f"{eval_result.get('macro_f1', 0):.4f}")
                            c3.metric("Weighted F1", f"{eval_result.get('weighted_f1', 0):.4f}")

                            per_class = eval_result.get("per_class_metrics", {})
                            if per_class:
                                pc_df = pd.DataFrame(per_class).T
                                st.dataframe(pc_df, use_container_width=True)
        else:
            st.info("No training experiments yet.")

elif page == "📈 Evaluation":
    st.title("📈 Evaluation & Comparison")

    tab_curve, tab_compare, tab_sig = st.tabs(["Learning Curve", "Strategy Comparison", "Significance Test"])

    with tab_curve:
        st.subheader("Low-Resource Learning Curve")
        datasets = api_get("/datasets") or []
        if datasets:
            all_versions = []
            for d in datasets:
                detail = api_get(f"/datasets/{d['id']}")
                if detail:
                    for v in detail.get("versions", []):
                        all_versions.append({"id": v["id"], "dataset_id": d["id"], "name": f"v{v['id']}: {d['name']}/{v['version_name']}"})

            if all_versions:
                v_opts = {v["name"]: v for v in all_versions}
                sel_v = st.selectbox("Version", list(v_opts.keys()))
                v_info = v_opts[sel_v]

                backbone = st.selectbox("Backbone", ["distilbert", "tinybert", "textcnn", "bilstm_attention"], key="lc_bb")
                mode = st.selectbox("Training Mode", ["baseline", "augmented"], key="lc_mode")
                fractions = st.multiselect("Data Fractions", [0.1, 0.2, 0.5, 1.0], default=[0.1, 0.2, 0.5, 1.0])

                if st.button("Run Learning Curve"):
                    payload = {
                        "dataset_id": v_info["dataset_id"],
                        "version_id": v_info["id"],
                        "backbone": backbone,
                        "training_mode": mode,
                        "data_fractions": fractions,
                        "hyperparams": {"learning_rate": 2e-5, "batch_size": 16, "epochs": 5, "early_stopping_patience": 3, "max_seq_length": 128},
                    }
                    with st.spinner("Running learning curve experiment..."):
                        result = api_post("/evaluation/learning-curve", data=payload, timeout=300)
                        if result and "results" in result:
                            lc_data = result["results"]
                            if lc_data:
                                lc_df = pd.DataFrame(lc_data)
                                fig = px.line(lc_df, x="fraction", y=["accuracy", "macro_f1"], markers=True,
                                              title="Learning Curve: Metrics vs Data Fraction")
                                st.plotly_chart(fig, use_container_width=True)
                                st.dataframe(lc_df, use_container_width=True)
                        elif result and "error" in result:
                            st.error(f"Error: {result['error']}")

    with tab_compare:
        st.subheader("Strategy Comparison")
        exps = api_get("/training/experiments") or []
        completed = [e for e in exps if e.get("status") == "completed"]
        if completed:
            exp_opts = {f"#{e['id']} {e['experiment_name']} ({e['training_mode']})": e["id"] for e in completed}
            selected = st.multiselect("Select Experiments to Compare", list(exp_opts.keys()))
            if st.button("Compare") and len(selected) >= 2:
                exp_ids = [exp_opts[s] for s in selected]
                result = api_post("/evaluation/compare-strategies", data={"version_ids": exp_ids})
                if result and "comparisons" in result:
                    cmp_df = pd.DataFrame(result["comparisons"])
                    st.dataframe(cmp_df, use_container_width=True)

                    fig = go.Figure()
                    for _, row in cmp_df.iterrows():
                        fig.add_trace(go.Bar(
                            name=row.get("experiment_name", str(row.get("experiment_id", ""))),
                            x=["Accuracy", "Macro F1", "Weighted F1"],
                            y=[row.get("accuracy", 0), row.get("macro_f1", 0), row.get("weighted_f1", 0)],
                        ))
                    fig.update_layout(title="Strategy Comparison", barmode="group")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No completed experiments to compare.")

    with tab_sig:
        st.subheader("Statistical Significance Test")
        exps = api_get("/training/experiments") or []
        completed = [e for e in exps if e.get("status") == "completed"]
        if len(completed) >= 2:
            exp_opts = {f"#{e['id']} {e['experiment_name']}": e["id"] for e in completed}
            c1, c2 = st.columns(2)
            with c1:
                sel_a = st.selectbox("Experiment A", list(exp_opts.keys()), key="sig_a")
            with c2:
                sel_b = st.selectbox("Experiment B", list(exp_opts.keys()), index=1, key="sig_b")

            test_type = st.selectbox("Test Type", ["paired_t", "bootstrap"])
            if st.button("Run Significance Test"):
                payload = {
                    "experiment_id_a": exp_opts[sel_a],
                    "experiment_id_b": exp_opts[sel_b],
                    "test_type": test_type,
                }
                result = api_post("/evaluation/significance-test", data=payload)
                if result:
                    st.json(result)
                    if result.get("significant"):
                        st.success("Difference is statistically significant! ✅")
                    else:
                        st.info("Difference is NOT statistically significant.")
        else:
            st.info("Need at least 2 completed experiments.")
