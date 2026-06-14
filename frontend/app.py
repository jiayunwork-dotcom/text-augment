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
    ["📊 Dashboard", "📁 Datasets", "🔧 Augmentation", "🔍 Filtering", "🏷️ Annotation", "🏋️ Training", "📈 Evaluation"],
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

elif page == "🏷️ Annotation":
    st.title("🏷️ Active Learning Annotation")

    tab_dashboard, tab_create, tab_annotate, tab_dispute, tab_consistency = st.tabs(
        ["📊 Dashboard", "➕ Create Queue", "✍️ Annotate", "⚖️ Dispute", "📏 Consistency"]
    )

    with tab_dashboard:
        st.subheader("Annotation Dashboard")
        queues = api_get("/annotation/queues") or []

        if queues:
            active_queues = [q for q in queues if q.get("status") in ("pending", "in_progress", "completed")]
            if active_queues:
                queue_opts = {
                    f"#{q['id']} {q['name']} ({q['status']}) - {q.get('progress', {}).get('total', 0)} items": q["id"]
                    for q in active_queues
                }
                sel_q = st.selectbox("Select Active Queue", list(queue_opts.keys()), key="dash_queue")
                if sel_q:
                    qid = queue_opts[sel_q]
                    queue_detail = api_get(f"/annotation/queues/{qid}")
                    if queue_detail:
                        prog = queue_detail.get("progress", {})
                        total = prog.get("total", 0)
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Total Items", total)
                        col2.metric("Pending", prog.get("pending", 0))
                        col3.metric("Annotated", prog.get("annotated", 0) + prog.get("arbitrated", 0))
                        col4.metric("Disputed", prog.get("disputed", 0))

                        finalized = prog.get("annotated", 0) + prog.get("arbitrated", 0)
                        confirm = prog.get("confirm_count", 0)
                        relabel = prog.get("relabel_count", 0)
                        discard = prog.get("discard_count", 0)
                        pending = prog.get("pending", 0) + prog.get("locked", 0)
                        disputed = prog.get("disputed", 0)

                        if total > 0:
                            c1, c2 = st.columns(2)
                            with c1:
                                pie_data = []
                                pie_labels = []
                                if confirm > 0:
                                    pie_data.append(confirm)
                                    pie_labels.append(f"Confirmed ({confirm})")
                                if relabel > 0:
                                    pie_data.append(relabel)
                                    pie_labels.append(f"Relabeled ({relabel})")
                                if discard > 0:
                                    pie_data.append(discard)
                                    pie_labels.append(f"Discarded ({discard})")
                                if pending > 0:
                                    pie_data.append(pending)
                                    pie_labels.append(f"Pending ({pending})")
                                if disputed > 0:
                                    pie_data.append(disputed)
                                    pie_labels.append(f"Disputed ({disputed})")

                                if pie_data:
                                    fig = go.Figure(data=[go.Pie(
                                        labels=pie_labels,
                                        values=pie_data,
                                        hole=0.4,
                                        marker=dict(colors=["#4CAF50", "#FFC107", "#F44336", "#9E9E9E", "#FF5722"])
                                    )])
                                    fig.update_layout(title="Annotation Status Distribution")
                                    st.plotly_chart(fig, use_container_width=True)

                            with c2:
                                if finalized > 0:
                                    dec_data = [confirm, relabel, discard]
                                    dec_labels = [f"Confirm {confirm/finalized:.1%}", f"Relabel {relabel/finalized:.1%}", f"Discard {discard/finalized:.1%}"]
                                    fig2 = go.Figure(data=[go.Bar(
                                        x=dec_data,
                                        y=dec_labels,
                                        orientation="h",
                                        marker=dict(color=["#4CAF50", "#FFC107", "#F44336"])
                                    )])
                                    fig2.update_layout(title="Decision Distribution (Finalized)")
                                    st.plotly_chart(fig2, use_container_width=True)

                        st.subheader("Queue Info")
                        st.json(queue_detail)

                        if queue_detail.get("status") == "completed":
                            st.success("✅ Queue completed! Ready to apply.")
                            applied_by = st.text_input("Applied by (your name)", value="admin", key="apply_by")
                            if st.button("🚀 Apply Annotation Results", type="primary"):
                                with st.spinner("Applying results to create annotated version..."):
                                    result = api_post("/annotation/apply", data={
                                        "queue_id": qid,
                                        "applied_by": applied_by,
                                    })
                                    if "error" in result:
                                        st.error(f"Failed: {result['error']}")
                                    else:
                                        st.success("✅ Applied successfully!")
                                        st.json(result)
            else:
                st.info("No active queues. Create one in the 'Create Queue' tab.")
        else:
            st.info("No queues yet. Create one in the 'Create Queue' tab.")

        st.subheader("All Queues")
        if queues:
            q_df = pd.DataFrame(queues)
            disp_cols = ["id", "name", "version_id", "status", "capacity", "review_mode", "num_reviewers", "created_at"]
            avail = [c for c in disp_cols if c in q_df.columns]
            st.dataframe(q_df[avail], use_container_width=True)

    with tab_create:
        st.subheader("Create Annotation Queue")
        datasets = api_get("/datasets") or []
        if not datasets:
            st.warning("No datasets available.")
        else:
            all_versions = []
            for d in datasets:
                detail = api_get(f"/datasets/{d['id']}")
                if detail:
                    for v in detail.get("versions", []):
                        if v.get("version_type") == "filtered":
                            all_versions.append({
                                "id": v["id"],
                                "name": f"v{v['id']}: {d['name']} / {v['version_name']} ({v.get('total_samples',0)} samples)",
                            })

            if not all_versions:
                st.warning("⚠️ No filtered versions available. Please run a quality filter task first.")
            else:
                v_opts = {v["name"]: v["id"] for v in all_versions}
                sel_v = st.selectbox("Select Filtered Version", list(v_opts.keys()), key="create_v")
                v_id = v_opts[sel_v]

                col1, col2 = st.columns(2)
                with col1:
                    q_name = st.text_input("Queue Name", value=f"queue_v{v_id}")
                    capacity = st.number_input("Capacity (max items)", 1, 10000, 100, 10)
                    lock_timeout = st.number_input("Lock Timeout (minutes)", 1, 1440, 30, 5)

                with col2:
                    review_mode = st.radio("Review Mode", ["single", "multi"], horizontal=True)
                    if review_mode == "multi":
                        num_reviewers = st.number_input("Number of Reviewers (odd)", 1, 9, 3, 2)
                        if num_reviewers % 2 == 0:
                            st.warning("⚠️ Multi-review mode requires odd number of reviewers.")
                    else:
                        num_reviewers = 1

                created_by = st.text_input("Created by", value="admin")

                if st.button("✨ Create Annotation Queue", type="primary"):
                    if review_mode == "multi" and num_reviewers % 2 == 0:
                        st.error("Number of reviewers must be odd for multi-review mode.")
                    else:
                        payload = {
                            "version_id": v_id,
                            "name": q_name,
                            "capacity": capacity,
                            "review_mode": review_mode,
                            "num_reviewers": num_reviewers,
                            "lock_timeout_minutes": lock_timeout,
                            "created_by": created_by,
                        }
                        result = api_post("/annotation/queues", data=payload)
                        if "error" in result:
                            st.error(f"Failed: {result['error']}")
                        else:
                            st.success(f"✅ Queue created! ID: {result.get('id')}, Items: {result.get('progress', {}).get('total', 0)}")
                            st.json(result)

    with tab_annotate:
        st.subheader("Annotation Workspace")
        queues = api_get("/annotation/queues") or []
        active = [q for q in queues if q.get("status") in ("pending", "in_progress")]

        if not active:
            st.info("No active queues for annotation. Create or wait for a queue.")
        else:
            annotator_id = st.text_input("👤 Annotator ID", value="annotator_1", key="ann_id")

            q_opts = {f"#{q['id']} {q['name']} ({q['status']})": q["id"] for q in active}
            sel_q = st.selectbox("Select Queue", list(q_opts.keys()), key="annotate_q")
            if sel_q:
                qid = q_opts[sel_q]
                batch_size = st.number_input("Batch Size", 1, 100, 5, 1, key="batch_size")

                if "current_batch" not in st.session_state:
                    st.session_state.current_batch = []
                if "batch_decisions" not in st.session_state:
                    st.session_state.batch_decisions = {}

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🎯 Claim Next Batch", type="primary"):
                        with st.spinner("Claiming tasks..."):
                            result = api_post("/annotation/claim", data={
                                "queue_id": qid,
                                "annotator_id": annotator_id,
                                "batch_size": batch_size,
                            })
                            if isinstance(result, list):
                                st.session_state.current_batch = result
                                st.session_state.batch_decisions = {}
                                if not result:
                                    st.info("No more tasks available to claim.")
                            elif isinstance(result, dict) and "error" in result:
                                st.error(f"Failed: {result['error']}")

                with col2:
                    if st.button("🔓 Release All Locks"):
                        item_ids = [s["item_id"] for s in st.session_state.current_batch]
                        if item_ids:
                            result = api_post("/annotation/release", data={
                                "queue_id": qid,
                                "annotator_id": annotator_id,
                                "item_ids": item_ids,
                            })
                            st.success(f"Released {result.get('released_count', 0)} locks")
                            st.session_state.current_batch = []
                            st.session_state.batch_decisions = {}
                        else:
                            st.info("No locks to release")

                if st.session_state.current_batch:
                    st.subheader(f"📝 Annotating Batch ({len(st.session_state.current_batch)} samples)")

                    for idx, sample in enumerate(st.session_state.current_batch):
                        item_id = sample["item_id"]
                        with st.container():
                            st.markdown(f"### Sample {idx + 1} (Item #{item_id})")
                            unc = sample.get("uncertainty_score", 0)
                            col_a, col_b, col_c = st.columns(3)
                            col_a.metric("Uncertainty", f"{unc:.3f}")
                            col_b.metric("Current Label", sample.get("current_label", "N/A"))
                            conf = sample.get("confidence")
                            col_c.metric("Model Confidence", f"{conf:.3f}" if conf else "N/A")

                            if sample.get("similarity_score") is not None:
                                st.caption(f"Similarity Score: {sample['similarity_score']:.3f} | "
                                           f"Perplexity: {sample.get('perplexity', 'N/A')}")

                            st.info(f"**Text:** {sample.get('text', '')}")

                            decision_key = f"dec_{item_id}"
                            label_key = f"lbl_{item_id}"
                            comment_key = f"cmt_{item_id}"

                            if decision_key not in st.session_state.batch_decisions:
                                st.session_state.batch_decisions[decision_key] = "confirm"

                            c1, c2 = st.columns(2)
                            with c1:
                                decision = st.radio(
                                    "Decision",
                                    ["confirm", "relabel", "discard"],
                                    index=["confirm", "relabel", "discard"].index(
                                        st.session_state.batch_decisions.get(decision_key, "confirm")
                                    ),
                                    key=f"radio_{item_id}",
                                    horizontal=True,
                                )
                                st.session_state.batch_decisions[decision_key] = decision

                            with c2:
                                if decision == "relabel":
                                    new_label = st.text_input(
                                        "New Label",
                                        value=st.session_state.batch_decisions.get(label_key, ""),
                                        key=f"input_{item_id}",
                                    )
                                    st.session_state.batch_decisions[label_key] = new_label

                            comment = st.text_area(
                                "Comment (optional)",
                                key=f"ta_{item_id}",
                                height=60,
                            )
                            st.session_state.batch_decisions[comment_key] = comment

                            st.divider()

                    if st.button("✅ Submit All Decisions", type="primary"):
                        items_payload = []
                        has_error = False
                        for sample in st.session_state.current_batch:
                            item_id = sample["item_id"]
                            dec = st.session_state.batch_decisions.get(f"dec_{item_id}", "confirm")
                            new_lbl = st.session_state.batch_decisions.get(f"lbl_{item_id}", "")
                            cmt = st.session_state.batch_decisions.get(f"cmt_{item_id}", "")

                            if dec == "relabel" and not new_lbl:
                                st.error(f"❌ Item #{item_id}: new_label required for relabel decision")
                                has_error = True
                                continue

                            item_payload = {
                                "item_id": item_id,
                                "decision": dec,
                                "new_label": new_lbl if dec == "relabel" else None,
                                "comment": cmt or None,
                            }
                            items_payload.append(item_payload)

                        if not has_error and items_payload:
                            with st.spinner("Submitting annotations..."):
                                result = api_post("/annotation/submit", data={
                                    "queue_id": qid,
                                    "annotator_id": annotator_id,
                                    "items": items_payload,
                                })
                                if "error" in str(result):
                                    st.error(f"Submission error: {result}")
                                else:
                                    processed = result.get("processed_count", 0)
                                    errors = result.get("errors", [])
                                    st.success(f"✅ Submitted {processed} items successfully!")
                                    if errors:
                                        st.warning(f"⚠️ {len(errors)} errors:")
                                        for e in errors:
                                            st.write(e)
                                    st.session_state.current_batch = []
                                    st.session_state.batch_decisions = {}

    with tab_dispute:
        st.subheader("⚖️ Dispute Resolution")
        queues = api_get("/annotation/queues") or []
        q_with_any = [q for q in queues if q.get("status") in ("pending", "in_progress", "completed")]

        if not q_with_any:
            st.info("No queues available.")
        else:
            q_opts = {f"#{q['id']} {q['name']}": q["id"] for q in q_with_any}
            sel_q = st.selectbox("Select Queue", list(q_opts.keys()), key="dispute_q")
            if sel_q:
                qid = q_opts[sel_q]
                arbitrator_id = st.text_input("👨‍⚖️ Arbitrator ID", value="admin", key="arb_id")

                disputed = api_get(f"/annotation/queues/{qid}/disputed") or []
                if not disputed:
                    st.success("✅ No disputed items in this queue.")
                else:
                    st.warning(f"⚠️ {len(disputed)} disputed item(s) found")

                    if "arb_decisions" not in st.session_state:
                        st.session_state.arb_decisions = {}

                    for d in disputed:
                        item_id = d["item_id"]
                        with st.expander(f"Disputed Item #{item_id} (Uncertainty: {d.get('uncertainty_score', 0):.3f})", expanded=True):
                            st.write(f"**Sample ID:** {d['sample_id']}")
                            st.write(f"**Current Label:** {d['current_label']}")
                            st.info(f"**Text:** {d['text']}")

                            st.subheader("Annotator Records")
                            records = d.get("records", [])
                            for r in records:
                                st.write(f"- **{r['annotator_id']}**: {r['decision']} "
                                         f"{'→ ' + r['new_label'] if r.get('new_label') else ''} "
                                         f"({r.get('created_at', '')})")
                                if r.get("comment"):
                                    st.caption(f"  Comment: {r['comment']}")

                            st.subheader("Arbitration Decision")
                            d_key = f"arb_dec_{item_id}"
                            l_key = f"arb_lbl_{item_id}"
                            c_key = f"arb_cmt_{item_id}"

                            if d_key not in st.session_state.arb_decisions:
                                st.session_state.arb_decisions[d_key] = "confirm"

                            dec = st.radio(
                                "Final Decision",
                                ["confirm", "relabel", "discard"],
                                index=["confirm", "relabel", "discard"].index(
                                    st.session_state.arb_decisions.get(d_key, "confirm")
                                ),
                                key=f"arb_radio_{item_id}",
                                horizontal=True,
                            )
                            st.session_state.arb_decisions[d_key] = dec

                            if dec == "relabel":
                                new_lbl = st.text_input(
                                    "New Label",
                                    value=st.session_state.arb_decisions.get(l_key, ""),
                                    key=f"arb_input_{item_id}",
                                )
                                st.session_state.arb_decisions[l_key] = new_lbl

                            cmt = st.text_area("Comment (optional)", key=f"arb_ta_{item_id}", height=60)
                            st.session_state.arb_decisions[c_key] = cmt

                    if st.button("⚖️ Submit Arbitration", type="primary"):
                        items_payload = []
                        has_error = False
                        for d in disputed:
                            item_id = d["item_id"]
                            dec = st.session_state.arb_decisions.get(f"arb_dec_{item_id}", "confirm")
                            new_lbl = st.session_state.arb_decisions.get(f"arb_lbl_{item_id}", "")
                            cmt = st.session_state.arb_decisions.get(f"arb_cmt_{item_id}", "")

                            if dec == "relabel" and not new_lbl:
                                st.error(f"❌ Item #{item_id}: new_label required for relabel")
                                has_error = True
                                continue

                            items_payload.append({
                                "item_id": item_id,
                                "decision": dec,
                                "new_label": new_lbl if dec == "relabel" else None,
                                "comment": cmt or None,
                            })

                        if not has_error and items_payload:
                            with st.spinner("Submitting arbitration..."):
                                result = api_post("/annotation/arbitrate", data={
                                    "queue_id": qid,
                                    "arbitrator_id": arbitrator_id,
                                    "items": items_payload,
                                })
                                if "error" in str(result):
                                    st.error(f"Error: {result}")
                                else:
                                    processed = result.get("processed_count", 0)
                                    errors = result.get("errors", [])
                                    st.success(f"✅ Arbitrated {processed} items!")
                                    if errors:
                                        st.warning(f"⚠️ {len(errors)} errors")
                                    st.session_state.arb_decisions = {}

    with tab_consistency:
        st.subheader("📏 Annotation Consistency Report")
        queues = api_get("/annotation/queues") or []
        q_with_ann = [q for q in queues if (q.get("progress", {}).get("annotated", 0) + (q.get("progress", {}).get("arbitrated", 0))) > 0]

        if not q_with_ann:
            st.info("No annotated queues available yet.")
        else:
            q_opts = {f"#{q['id']} {q['name']}": q["id"] for q in q_with_ann}
            sel_q = st.selectbox("Select Queue", list(q_opts.keys()), key="cons_q")
            if sel_q and st.button("📊 Generate Report", type="primary"):
                qid = q_opts[sel_q]
                report = api_get(f"/annotation/queues/{qid}/consistency")

                if report:
                    kappa = report.get("cohens_kappa", 0.0)
                    level = report.get("kappa_level", "")

                    if report.get("warning"):
                        st.warning(f"⚠️ {report['warning']}")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Cohen's Kappa", f"{kappa:.4f}")
                    level_labels = {
                        "excellent": ("🌟 Excellent", "#4CAF50"),
                        "good": ("✅ Good", "#2196F3"),
                        "needs_attention": ("⚠️ Needs Attention", "#FF9800"),
                    }
                    label_text, color = level_labels.get(level, ("Unknown", "#9E9E9E"))
                    col2.markdown(f"<h3 style='color: {color};'>{label_text}</h3>", unsafe_allow_html=True)
                    col3.metric("Kappa Threshold", "≥ 0.6")

                    st.write("**Kappa Level Guide:**")
                    st.write("- 🟢 ≥ 0.8: Excellent agreement")
                    st.write("- 🔵 0.6 ~ 0.8: Good agreement")
                    st.write("- 🟠 < 0.6: Needs attention / improvement")

                    level_order = ["excellent", "good", "needs_attention"]
                    level_colors = ["#4CAF50", "#2196F3", "#FF9800"]
                    kappa_bar_fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=kappa,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Cohen's Kappa"},
                        delta={'reference': 0.6},
                        gauge={
                            'axis': {'range': [0, 1]},
                            'bar': {'color': level_colors[level_order.index(level)] if level in level_order else "#9E9E9E"},
                            'steps': [
                                {'range': [0, 0.6], 'color': '#FFEBEE'},
                                {'range': [0.6, 0.8], 'color': '#E3F2FD'},
                                {'range': [0.8, 1.0], 'color': '#E8F5E9'},
                            ],
                            'threshold': {
                                'line': {'color': '#F44336', 'width': 4},
                                'thickness': 0.75,
                                'value': 0.6,
                            },
                        },
                    ))
                    st.plotly_chart(kappa_bar_fig, use_container_width=True)

                    pairwise = report.get("pairwise_kappa", {})
                    if pairwise:
                        st.subheader("👥 Pairwise Kappa (Between Annotators)")
                        pw_df = pd.DataFrame(
                            [(k.replace("_vs_", " ↔ "), v) for k, v in pairwise.items()],
                            columns=["Annotator Pair", "Cohen's Kappa"]
                        )
                        st.dataframe(pw_df, use_container_width=True)

                    ann_stats = report.get("annotator_stats", [])
                    if ann_stats:
                        st.subheader("📋 Annotator Statistics")
                        as_df = pd.DataFrame(ann_stats)
                        st.dataframe(as_df, use_container_width=True)

                        stats_fig = go.Figure()
                        for s in ann_stats:
                            stats_fig.add_trace(go.Bar(
                                name=s["annotator_id"],
                                x=["Confirm", "Relabel", "Discard"],
                                y=[s["confirm_count"], s["relabel_count"], s["discard_count"]],
                            ))
                        stats_fig.update_layout(
                            title="Annotator Decisions by Annotator",
                            barmode="group",
                        )
                        st.plotly_chart(stats_fig, use_container_width=True)

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
                            if vtype == "filtered":
                                status_icon = "✅"
                            elif vtype == "annotated":
                                status_icon = "🏷️"
                            elif vtype == "augmented":
                                status_icon = "⚠️"
                            else:
                                status_icon = "⚠️"
                            all_versions.append({
                                "id": v["id"],
                                "dataset_id": d["id"],
                                "version_type": vtype,
                                "name": f"{status_icon} v{v['id']}: {d['name']} / {v['version_name']} [{vtype}] ({v.get('total_samples',0)} samples)",
                            })

            exp_name = st.text_input("Experiment Name", value="experiment_1")

            if all_versions:
                filtered_only = [v for v in all_versions if v["version_type"] in ("filtered", "annotated")]
                non_filtered = [v for v in all_versions if v["version_type"] not in ("filtered", "annotated")]

                if filtered_only:
                    st.success(f"✅ {len(filtered_only)} filtered/annotated version(s) available for training")
                if non_filtered:
                    st.warning(f"⚠️ {len(non_filtered)} version(s) not yet filtered/annotated - cannot be used for training")

                v_opts = {v["name"]: v["id"] for v in all_versions}
                sel_v = st.selectbox("Dataset Version", list(v_opts.keys()))
                v_id = v_opts[sel_v]
                sel_version = next(v for v in all_versions if v["id"] == v_id)
                sel_ds_id = next(v["dataset_id"] for v in all_versions if v["id"] == v_id)

                if sel_version["version_type"] not in ("filtered", "annotated"):
                    st.error(f"❌ Selected version is of type '{sel_version['version_type']}'. "
                             "Only 'filtered' or 'annotated' versions can be used for training. "
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
