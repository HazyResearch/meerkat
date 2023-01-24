<script lang="ts">
	export let message: string;
	export let name: string;
	export let time: string;
</script>

<div class="flex ml-2 pl-4 pr-4 mb-4">
	<slot name="avatar" />
	<div class="ml-2 w-full">
		<p class="flex justify-between">
			<slot name="name">
				<span class="font-bold">{name}</span>
			</slot>
			<slot name="time">
				<span class="text-gray-500">{time}</span>
			</slot>
		</p>
		<p class="text-gray-800 whitespace-pre-line">
			{message}
		</p>
	</div>
</div>
