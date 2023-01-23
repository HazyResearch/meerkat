<script lang="ts">
	import { get_rows } from '$lib/api/api';
	import type { DataFrameRef } from '$lib/api/dataframe';
	import { Avatar, Button, Textarea } from 'flowbite-svelte';
	import { createEventDispatcher } from 'svelte';
	import Message from './Message.svelte';
	const eventDispatcher = createEventDispatcher();

	export let df: DataFrameRef;
	export let imgChatbot: string;
	export let imgUser: string;

	$: messages_promise = get_rows(df.ref_id, 0, 100000);

	let value: string = '';
	let send = () => {
		eventDispatcher('send', { message: value });
		value = '';
	};
</script>

<div class="bg-violet-100 p-4 rounded-lg flex flex-col h-full justify-between shadow-md">
	<div class="flex flex-col-reverse overflow-y-scroll h-full">
		{#await messages_promise then messages}
			{#each messages.rows as _, i}
				<Message
					message={messages.get_cell(messages.full_length - i - 1, 'message').data}
					name={messages.get_cell(messages.full_length - i - 1, 'name').data}
					time={messages.get_cell(messages.full_length - i - 1, 'time').data}
				>
					<svelte:fragment slot="avatar">
						<Avatar
							src={messages.get_cell(messages.full_length - i - 1, 'sender').data === 'chatbot'
								? imgChatbot
								: imgUser}
							stacked={true}
							class="mr-2"
						/>
					</svelte:fragment>
				</Message>
			{/each}
		{/await}
	</div>
	<div class="flex mt-4">
		<Textarea
			rows="5"
			placeholder="Your message..."
			class="mx-4 resize-none"
			bind:value
			on:keyup={(e) => {
				e.key === 'Enter' && !e.shiftKey ? send() : null;
			}}
		/>
		<Button class="text-violet-600 dark:text-violet-500" on:click={send} color="light">
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
