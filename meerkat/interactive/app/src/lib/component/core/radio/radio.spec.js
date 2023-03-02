import { fireEvent, render, screen } from '@testing-library/svelte';
import { describe, expect, it, vi } from 'vitest';
import Radio from './Radio.svelte';
import RadioGroup from './RadioGroup.svelte';

describe('Radio', () => {
    it('should render Radio', async () => {
        // Render a couple of radio buttons
        const { container: container_1, component: component_1 } = render(Radio, {
            props: {
                name: 'test',
                value: '1',
            },
        });

        const { container: container_2, component: component_2 } = render(Radio, {
            props: {
                name: 'test',
                value: '2',
            },
        });

        expect(container_1).toMatchSnapshot();
        expect(container_2).toMatchSnapshot();

        // Check that the radios are rendered
        const radios = screen.getAllByRole('radio');
        expect(radios).toHaveLength(2);

        // Mock the click event on the first radio
        const change = vi.fn();
        component_1.$on(`change`, change);

        // Click the radio
        await fireEvent.click(radios[0]);

        // Check the send event was fired
        expect(change).toHaveBeenCalledTimes(1);

        // Check that the second radio is not checked
        expect(radios[1]).not.toBeChecked();
    });
});


describe('RadioGroup', () => {
    it('should render RadioGroup', async () => {
        const { container, component } = render(RadioGroup, {
            props: {
                values: ['1', '2', '3'],
                selected: 0,
            },
        });

        // Check the radio group is rendered
        expect(container).toMatchSnapshot();

        // Check that the radios are rendered
        const radios = screen.getAllByRole('radio');
        expect(radios).toHaveLength(3);

        // Check that the first radio is checked
        expect(radios[0]).toBeChecked();

        // Mock the click event on the second radio
        const change = vi.fn();
        component.$on(`change`, change);

        // Click the second radio
        await fireEvent.click(radios[1]);

        // Check the send event was fired
        expect(change).toHaveBeenCalledTimes(1);

        // Check that the second radio is checked
        expect(radios[1]).toBeChecked();
    });
});
