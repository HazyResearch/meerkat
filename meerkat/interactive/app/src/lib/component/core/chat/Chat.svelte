<script lang="ts">
	import { fetchChunk } from '$lib/utils/api';
	import { DataFrameChunk, type DataFrameRef } from '$lib/utils/dataframe';
	import { Avatar, Button, Textarea } from 'flowbite-svelte';
	import { createEventDispatcher } from 'svelte';
	import { writable, type Writable } from 'svelte/store';
	import Message from './Message.svelte';

	const eventDispatcher = createEventDispatcher();

	export let df: DataFrameRef;
	export let imgChatbot: string;
	export let imgUser: string;

	let messages: Writable<DataFrameChunk> = writable(new DataFrameChunk([], [], [], 0, 'pkey'));
	let value: string = '';

	$: fetchChunk({ df: df, start: 0 }).then((newChunk) => messages.set(newChunk));

	let textarea = null;
	// Reset the textarea value when the response is received i.e. df was updated.
	let resetValue = () => {
		value = '';
		if (textarea !== null) {
			// Make the textarea active and editable again.
			textarea.focus();
			textarea.removeAttribute('readonly');
		}
	};
	$: df, resetValue();

	let sendMessage = () => {
		// Remove the newline character at the end of the message if it exists.
		if (value[value.length - 1] === '\n') {
			value = value.slice(0, -1);
		}
		eventDispatcher('send', { message: value });
		if (document.activeElement instanceof HTMLElement) {
			// Deactivate the textarea so that the user can't send the same message again.
			textarea = document.activeElement;
			document.activeElement.blur();
			// Make the textarea non-editable.
			textarea.setAttribute('readonly', 'readonly');
		}
	};
</script>

<div
	class="h-full bg-gray-50 dark:bg-slate-300 p-4 rounded-lg flex flex-col justify-between shadow-md"
>
	<div class="flex flex-col-reverse overflow-y-scroll">
		{#each $messages.rows as _, i}
			<Message
				message={$messages.getCell($messages.fullLength - i - 1, 'message').data}
				name={$messages.getCell($messages.fullLength - i - 1, 'name').data}
				time={$messages.getCell($messages.fullLength - i - 1, 'time').data}
			>
				<svelte:fragment slot="avatar">
					<Avatar
						src={$messages.getCell($messages.fullLength - i - 1, 'sender').data === 'chatbot'
							? imgChatbot
							: imgUser}
						stacked={true}
						class="mr-2"
					/>
				</svelte:fragment>
			</Message>
		{/each}
	</div>
	<div class="flex mt-4">
		<Textarea
			rows="5"
			placeholder="Your message..."
			class="mx-4 resize-none"
			bind:value
			on:keyup={(e) => {
				if (e.key === 'Enter' && !e.shiftKey) {
					sendMessage();
				}
			}}
		/>
		<Button class="text-violet-600 dark:text-violet-500" on:click={sendMessage} color="light">
			<svg fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-6 h-6">
				<path
					stroke-linecap="round"
					stroke-linejoin="round"
					d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5"
				/>
			</svg>
		</Button>
	</div>
</div>
