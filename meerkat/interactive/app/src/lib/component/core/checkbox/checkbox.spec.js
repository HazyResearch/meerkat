import { fireEvent, render, screen } from '@testing-library/svelte';
import { describe, expect, it, vi } from 'vitest';
import Checkbox from './Checkbox.svelte';

describe('Checkbox', () => {
	it('should render checkbox', async () => {

        // Ignore the TypeError: cellEdit is not a function
        // Render the checkbox
		const { container, component } = render(Checkbox, {
			props: {
                checked: true,
                color: 'red',
            },
		});

        // Check the checkbox is rendered
		expect(container).toMatchSnapshot();

        // Check the checkbox is rendered
        const checkbox = screen.getByRole('checkbox');
        expect(checkbox).toBeInTheDocument();

        // Mock the click event
		const change = vi.fn();
		component.$on(`change`, change);

        // Click the checkbox
        await fireEvent.click(checkbox);

		// Check the send event was fired
		// expect(change).toHaveBeenCalledTimes(1);
	});
});