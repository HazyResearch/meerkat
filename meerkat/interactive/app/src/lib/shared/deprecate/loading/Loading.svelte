<script lang="ts">
	import { BarLoader } from 'svelte-loading-spinners';
    import banner from "$lib/assets/banner_small.png"

</script>

<div class="flex justify-center h-full items-center">
	<img src={banner} alt="Meerkat" class="h-12"/>
    <BarLoader size="80" color="#7c3aed" unit="px" duration="1s" />
</div>
