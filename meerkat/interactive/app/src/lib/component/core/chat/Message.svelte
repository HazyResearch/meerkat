<script lang="ts">
	import CopyButton from '../copy_button/CopyButton.svelte';
	import Markdown from '../markdown/Markdown.svelte';

	export let message: string;
	export let name: string;
	export let time: string;

	let getBlocks = (message: str) => {
		// Split up message into blocks of text and code.
		let blocks: Array<{ type: 'text' | 'code'; value: string }> = [];
		let block: { type: 'text' | 'code'; value: string } = { type: 'text', value: '' };

		// Code blocks are delimited by triple backticks.
		let codeBlock = false;
		for (let i = 0; i < message.length; i++) {
			if (message[i] === '`' && message[i + 1] === '`' && message[i + 2] === '`') {
				if (codeBlock) {
					// Add the closing triple backticks.
					block.value += '```';
					blocks.push(block);
					block = { type: 'text', value: '' };
				} else {
					blocks.push(block);
					block = { type: 'code', value: '```' };
				}
				codeBlock = !codeBlock;
				i += 2;
			} else {
				block.value += message[i];
			}
		}
		blocks.push(block);
		return blocks;
	};

	$: blocks = getBlocks(message);
</script>

<div class="flex ml-2 pl-6 pr-4 pb-1 pt-1">
	<slot name="avatar" />
	<div class="ml-2 w-full">
		<p class="flex">
			<slot name="name">
				<span class="font-bold">{name}</span>
			</slot>
			<slot name="time">
				<span class="ml-auto text-gray-500">{time}</span>
			</slot>
			<span class="ml-2 self-center"><CopyButton bind:value={message} /></span>
		</p>
		<p class="text-gray-800 select-text flex flex-col">
			<!-- Show all the blocks, and add a CopyButton to 
				each code block in the margin. -->
			{#each blocks as block}
				{#if block.type === 'code'}
					<p class="flex justify-end relative top-6 right-2 fill-white">
						<CopyButton bind:value={block.value} />
					</p>
					<!-- Use max-w-full to fill out the width. 
					Must use grid, otherwise the code block will
					make the container too wide.
					-->
					<Markdown bind:body={block.value} classes="pb-2 max-w-full grid" />
				{:else}
					<Markdown bind:body={block.value} classes="max-w-full grid" />
				{/if}
			{/each}
		</p>
	</div>
</div>
