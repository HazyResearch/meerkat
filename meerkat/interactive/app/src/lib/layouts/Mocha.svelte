<script>
    import Plot from "$lib/component/plot/Plot.svelte";
    import Gallery from "$lib/component/gallery/Gallery.svelte";
    import Stats from "$lib/component/stats/Stats.svelte";

	export let components;

    let stats_component = components.stats;
	let gallery_component = components.gallery;
	let plot_component = components.plot;
    console.log(stats_component)
</script>


<div class="grid grid-cols-[1fr_2fr] py-10 h-screen gap-5">
    <div>
        <Stats {...stats_component.props}/>
        <Plot {...plot_component.props} />
    </div>
    <Gallery {...gallery_component.props} /> 
</div>

