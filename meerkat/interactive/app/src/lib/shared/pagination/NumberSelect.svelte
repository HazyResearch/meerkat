<script lang="ts">
	import Select from 'svelte-select';

	export let selection: { value: any; label: any } = { value: undefined, label: undefined };
	export let options: Array<{ value: any; label: number }>;
	export let hint_text: string = 'Text';

	function handleSelect(event: any) {
		selection = event.detail;
	}
</script>

<div class="flex">
	<div
		class="inline-flex self-center justify-center w-9/10 px-2 text-sm rounded-l-lg dark:bg-gray-800 dark:text-gray-100"
	>
		{hint_text}
	</div>
	<div class="self-center themed">
		<Select 
			value={selection.value}
			items={options}
			placeholder={hint_text}
			isClearable={false}
			showIndicator={false}
			on:select={handleSelect}
		/>
	</div>
</div>

<style>
	/* 	
			CSS variables can be used to control theming.
			https://github.com/rob-balfre/svelte-select/blob/master/docs/theming_variables.md
	*/

	.themed {
		--tw-bg-opacity: 1;
		--background: theme(colors.slate.200);
		--borderHoverColor: theme(colors.violet.600);
		--borderFocusColor: theme(colors.violet.600);
		--itemColor: theme(colors.gray.600);
		--itemHoverColor: theme(colors.gray.600);
		--itemActiveBackground: theme(colors.gray.800);
		--listBackground: theme(colors.gray.400);
		--listLeft: 0;
		--itemHoverBG: theme(colors.gray.200);
		--itemIsActiveBG: theme(colors.violet.600);
		--padding: 5px;
		--itemPadding: 2px;
		--selectedItemPadding: 0px;
		--height: 6px;
	}
</style>
