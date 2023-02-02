import { fireEvent, render, screen } from '@testing-library/svelte';
import { describe, expect, it, vi } from 'vitest';
import Choice from './Choice.svelte';

describe('Choice', () => {
	it('should render choice', async () => {
		const { container, component } = render(Choice, {
			props: {
                choices: ['A', 'B', 'C'],
                value: 'A',
                gui_type: 'radio',
                title: 'Title',
            },
		});
		expect(container).toMatchSnapshot();

        

        // // Check the button is rendered
        // const button = screen.getByRole('button');
        // expect(button).toBeInTheDocument();
        // expect(button).toHaveTextContent('hello');

        // // Mock the click event
		// const click = vi.fn();
		// component.$on(`click`, click);

        // // Click the button
        // await fireEvent.click(button);

		// // Check the send event was fired
		// expect(click).toHaveBeenCalledTimes(1);
	});
});