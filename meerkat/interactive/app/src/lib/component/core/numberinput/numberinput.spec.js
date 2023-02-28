import { fireEvent, render, screen } from '@testing-library/svelte';
import { describe, expect, it, vi } from 'vitest';
import NumberInput from './NumberInput.svelte';

describe('NumberInput', () => {
	it('should render NumberInput', async () => {
		const { container, component } = render(NumberInput, {
			props: {
                value: 1.3,
			},
		});
		expect(container).toMatchSnapshot();

		// Check the number input is rendered
        const numberinput = screen.getByRole('spinbutton');
        expect(numberinput).toBeInTheDocument();
        
        // Find `1.3` in the component
        console.log(component.$$.ctx)
        expect(component.$$.ctx).toContain(1.3);
	});
});