<script lang="ts">
	// Source: https://github.com/aknakos/sveltekit-plotly
	import { PlotlyLib } from '../store';

	import { createEventDispatcher, onMount } from 'svelte';
	const dispatch = createEventDispatcher();

	export let id: string = 'plot-' + Math.floor(Math.random() * 100).toString();
	export let data: Array<Record<string, any>>;
	export let layout: Record<string, any> = {};
	export let config: Record<string, any> = {};
	export let loaded: boolean = false;
	export let reloadPlot = 0;

	function init() {
		if (!$PlotlyLib) $PlotlyLib = window?.Plotly;
	}

	onMount(async () => init());

	const onHover = (d) => dispatch('hover', d);
	const onUnhover = (d) => dispatch('unhover', d);
	const onClick = (d) => dispatch('click', d);
	const onSelected = (d) => dispatch('selected', d);
	const onRelayout = (d) => dispatch('relayout', d);

	const generatePlot = (node, data, layout, config, reloadPlot) => {
		return $PlotlyLib.newPlot(node, data, { ...layout }, { ...config }).then(() => {
			node.on('plotly_hover', onHover);
			node.on('plotly_unhover', onUnhover);
			node.on('plotly_click', onClick);
			node.on('plotly_selected', onSelected);
			node.on('plotly_relayout', onRelayout);
			loaded = true;
		});
	};

	const updatePlot = (node, data, layout, config, reloadPlot) => {
		return $PlotlyLib.react(node, data, layout, config).then(() => {
			console.debug('update ploty', data);
			loaded = true;
		});
	};

	function plotlyAction(node, { data, layout, config, reloadPlot }) {
		generatePlot(node, data, layout, config, reloadPlot);

		return {
			update({ data, layout, config }) {
				loaded = false;
				updatePlot(node, data, layout, config, reloadPlot);
			},
			destroy() {
				node.removeListener('plotly_hover', onHover);
				node.removeListener('plotly_unhover', onUnhover);
				node.removeListener('plotly_click', onClick);
				node.removeListener('plotly_selected', onSelected);
				node.removeListener('plotly_relayout', onRelayout);
				loaded = false;
			}
		};
	}
</script>

<svelte:head>
	<script src="https://cdn.plot.ly/plotly-2.12.1.min.js" on:load={init}></script>
</svelte:head>

{#if $PlotlyLib}
	<div {id} use:plotlyAction={{ data, layout, config, reloadPlot }} {...$$restProps} />
{:else}
	<slot><div class="text-center bg-purple-50">Loading Plotly</div></slot>
{/if}
