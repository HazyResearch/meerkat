import { fireEvent, render, screen } from '@testing-library/svelte';
import { describe, expect, it, vi } from 'vitest';
import Chat from './Chat.svelte';

describe('Chat', () => {
	it('should render chat', async () => {
		const { container, component } = render(Chat, {
			props: {
				df: { refId: 'mock' },
				imgChatbot: 'https://placeimg.com/200/200/animals',
				imgUser: 'https://placeimg.com/200/200/people',
			},
		});

		// Check the messages in the chat are rendered
		expect(await screen.findByText('hello')).toBeTruthy();
		expect(container).toMatchSnapshot();

		// Mock the send event
		const send = vi.fn();
		component.$on(`send`, send);

		// Set the value of the textarea
		const textarea = screen.getByRole('textbox');
		textarea.setAttribute('value', 'blah blah');

		// Click the send button
		const button = screen.getByRole('button');
		await fireEvent.click(button);

		// Check the send event was fired
		expect(send).toHaveBeenCalledTimes(1);
	});
});