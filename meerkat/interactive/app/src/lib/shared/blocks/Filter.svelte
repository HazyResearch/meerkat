<script lang="ts">
import type { FilterCriterion } from "$lib/api/dataframe";

    import { getContext } from "svelte";
    
    const { dispatch, get_ref } = getContext('Block');

    export let df_ref_id: string;
    // let alias: string = 'df';

    let filter_args = {
        "filter-1": {"column": "label", "op": "==", "value": 0},
        "filter-2": {"column": "time", "op": "<", "value": 107},
        "filter-3": {"column": "color", "op": ">=", "value": 11},
    }

    let filter_undos = {
        "filter-1": () => {},
        "filter-2": () => {},
        "filter-3": () => {},
    }

    let handle_click = async (e) => {
        const filter_id: string = e.target.id;
        const checked = e.target.checked;
        const filter_info = filter_args[filter_id]
        // TODO (arjundd): Should filter_criteria pass in all filters that are active?
        const filter_criteria: Array<FilterCriterion> = [
            {
                "column": filter_info.column,
                "op": filter_info.op,
                "value": filter_info.value
            }
        ]

        // Dispatch filter when it's checked
        let return_value;
        if (checked) {
            return_value = await dispatch("filter", df_ref_id, filter_criteria);
            filter_undos[filter_id] = return_value.undo;
            console.log(return_value.undo);
            console.log(filter_undos[filter_id]);
            console.log("stored the undo fn");
        } else {
            console.log("Calling the unfilter");
            console.log(filter_undos[filter_id]);
            return_value = filter_undos[filter_id]();
            // dispatch({ type: 'filter', payload: { filter, df_ref_id, remove: true } });
        }
        // Dispatch unfilter when it's unchecked
        dispatch('filter', df_ref_id, filter);
    }
</script>

<div class="bg-slate-100">
    <input id="filter-1" type=checkbox on:change={handle_click} />Filter 1
    <input id="filter-2" type=checkbox on:change={handle_click} />Filter 2
    <input id="filter-3" type=checkbox on:change={handle_click} />Filter 3
</div>