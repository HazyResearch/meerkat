<script lang="ts">
    import RowModal from '$lib/shared/modals/RowModal.svelte';
	import { dispatch } from '$lib/utils/api';
	import type { Endpoint } from '$lib/utils/types';
	import { createEventDispatcher } from 'svelte';
	import Plot from '../plot/Plot.svelte';
    import { openModal } from 'svelte-modals';

	const eventDispatcher = createEventDispatcher();

	export let keyidxs: Array<string | number>;
    export let filteredDf: string;
	export let data: string;
    export let layout: string;
	export let onRelayout: Endpoint;
	export let selected: Array<number> = [];
    console.log(filteredDf)

    const open_row_modal = (posidx: number) => {
		openModal(RowModal, {
			df: filteredDf,
			posidx: posidx,
			mainColumn: 'text'
		});
	};


	async function onEndpoint(endpoint: Endpoint, e) {
		if (!endpoint) return;
		dispatch(endpoint.endpointId, {
			detail: { keyidxs: e.detail.points.map((p) => keyidxs[p.pointIndex]) }
		});
	}

    async function onClick(e) {
        console.log("click", e)
        open_row_modal(e.detail.points[0].pointIndex)
    }

	async function onSelected(e) {
        console.log(e)
		selected = e.detail ? e.detail.points.map((p) => keyidxs[p.pointIndex]) : [];
		eventDispatcher('select', { selected: selected });
	}

	async function dispatchRelayout(e) {
		if (!onRelayout) return;
		dispatch(onRelayout.endpointId, {
			detail: {
				x_range: [e.detail['xaxis.range[0]'], e.detail['xaxis.range[1]']],
                y_range: [e.detail['yaxis.range[0]'], e.detail['yaxis.range[1]']]
			}
		});
	}
    let layout_json = JSON.parse(layout);
    layout_json["plot_bgcolor"] = "rgba(0,0,0,0)"
    let config = {'displayModeBar': false}
</script>

<Plot
    layout={layout_json}
    data={JSON.parse(data)}
    config={config}
	on:click={(e) => onClick(e)}
	on:selected={(e) => onSelected(e)}
	on:relayout={(e) => dispatchRelayout(e)}
/>
