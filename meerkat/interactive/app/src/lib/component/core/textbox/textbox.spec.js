import { render, screen } from '@testing-library/svelte';
import { describe, expect, it } from 'vitest';
import Textbox from './Textbox.svelte';

describe('Textbox', () => {
	it('should render Textbox', async () => {
		const { container, component } = render(Textbox, {
			props: {
                text: 'hello',
			},
		});
		expect(container).toMatchSnapshot();

		// Check the textbox is rendered
        const textbox = screen.getByRole('textbox');
        expect(textbox).toBeInTheDocument();
        
        // Find `hello` in the component
        expect(component.$$.ctx).toContain('hello');

        // // Mock keyboard presses to type 'world'
        // const keydown = vi.fn();
        // component.$on(`keydown`, keydown);
        // const keyup = vi.fn();
        // component.$on(`keyup`, keyup);
        
        // // Type 'world'
        // await fireEvent.keyDown(textbox, { key: 'w' });
        // await fireEvent.keyUp(textbox, { key: 'w' });
        // await fireEvent.keyDown(textbox, { key: 'o' });
        // await fireEvent.keyUp(textbox, { key: 'o' });
        // await fireEvent.keyDown(textbox, { key: 'r' });
        // await fireEvent.keyUp(textbox, { key: 'r' });
        // await fireEvent.keyDown(textbox, { key: 'l' });
        // await fireEvent.keyUp(textbox, { key: 'l' });
        // await fireEvent.keyDown(textbox, { key: 'd' });
        // await fireEvent.keyUp(textbox, { key: 'd' });

        // // Check the send event was fired
        // expect(keydown).toHaveBeenCalledTimes(5);
        // expect(keyup).toHaveBeenCalledTimes(5);
	});
});