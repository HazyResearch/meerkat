import { fireEvent, render, screen } from '@testing-library/svelte';
import { describe, expect, it, vi } from 'vitest';
import Toggle from './Toggle.svelte';

describe('Toggle', () => {
	it('should render Toggle', async () => {
		const { container, component } = render(Toggle, {
			props: {
                value: true,
			},
		});
		expect(container).toMatchSnapshot();

		// Check the toggle is rendered
        const toggle = screen.getByRole('checkbox');
        expect(toggle).toBeInTheDocument();

        // Mock the click event
        const click = vi.fn();
        component.$on(`change`, click);

        // Click the toggle
        await fireEvent.click(toggle);

        // Check the send event was fired
        expect(click).toHaveBeenCalledTimes(1);
	});
});