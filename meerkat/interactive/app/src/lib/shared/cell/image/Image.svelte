<script lang="ts">
	export let data: string;
	export let height: string = "100%";
	export let width: string = "100%";
	export let layout: string = "object-cover";
</script>

<img 
	class="rounded-md shadow-md" 
	class:object-cover={layout === 'object-cover'}
	style:height={height} 
	style:width={width} 
	src={data} 
	alt="A cell." 
/>
