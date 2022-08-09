<script lang="ts">
	export let data: string;
	export let height: string = "100%";
	export let width: string = "100%";
	export let layout: string = "object-cover";
</script>

<img 
	class="rounded-sm" 
	class:object-cover={layout === 'object-cover'}
	style:height={height} 
	style:width={width} 
	src={data} 
	alt="A cell." 
/>
