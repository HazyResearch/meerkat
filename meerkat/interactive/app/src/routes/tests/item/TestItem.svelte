<script lang="ts">
	import Item from '$lib/components/item/Item.svelte';
</script>

<div class="flex bg-slate-50">
	<div class="basis-1/4 flex place-items-center justify-center bg-slate-500 text-slate-300">
		<Item data="Hello World" />
	</div>
	<div class="basis-1/4 flex place-items-center justify-center bg-slate-300 text-slate-500">
		<Item data="123" />
	</div>
	<div class="basis-1/2 flex m-2 place-items-start">
		<Item
			data="https://i.natgeofe.com/k/830b5d15-92db-429f-a80a-cc89b5700af5/mt-everest.jpg?w=636&h=437"
		/>
	</div>
</div>
