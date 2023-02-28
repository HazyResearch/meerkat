import { fireEvent, render, screen } from '@testing-library/svelte';
import { describe, expect, it, vi } from 'vitest';
import Button from './Button.svelte';

describe('Button', () => {
	it('should render button', async () => {
		const { container, component } = render(Button, {
			props: {
				title: 'hello',
			},
		});
		expect(container).toMatchSnapshot();

		// Check the button is rendered
		const button = screen.getByRole('button');
		expect(button).toBeInTheDocument();
		expect(button).toHaveTextContent('hello');

		// Mock the click event
		const click = vi.fn();
		component.$on(`click`, click);

		// Click the button
		await fireEvent.click(button);

		// Check the send event was fired
		// expect(click).toHaveBeenCalledTimes(1);
	});
});