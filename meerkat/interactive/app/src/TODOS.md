# Fixes
- fix base_datapanel_id bug: on changing app code and live reloading, selection of points on the plot does not work because the base_datpanel_id corresponds to a DP that is no longer available on the backend

# Structure


`interface` entrypoint


Modifier | Views


DataState Object
---------------
curent_datapanel_id
current_schema
...



View (datapanel_id) --> visual
    - Visualize DataState object

Modifier (datapanel_id) --> New_Object(new_datapanel_id | new_column)
    - Update DataState object

    - Match
      - Input: MatchSomething object (search text, search column)
      - Update DataState object
        - New schema (inplace) -- with the new column
        - New datapanel id -- with the new schema and sorted by the new column
        - Inplace vs. outofplace doesn't realy matter, just need the latest datapanel ID and schema


Maintain a sequence of applied Modifiers

3 Blocks
    - Plot
    - Match
    - Gallery

-- Match empty, plot has all points, gallery
-- Run match, sorts gallery, adds a column, plot is unchanged visually
-- Change plot x axis, select the new match column, select subset of points on plot

Classes of Modifiers:
    Views respond differently to different types of modifiers

1. Operations in the GUI form a linear chain from the original DP.
2. There are Operators and Viewers that are not mutually exclusive (Operators can be Viewers and Viewers can contain Operators).
   1. There is a notion of a Block, each Block is tied to a single DP.
   2. Operators and Viewers (or combinations thereof) are Blocks.
3. When an Operator operates on its DP, that Operation is reflected in all Blocks tied to the same DP.
4. An Interface is a collection of blocks. All blocks tied to the same DP are called a BlockGroup.
   1. We can create new blocks that disassociate from the BlockGroup to which they would belong. This can be used to implement a static view.
5. Operations include standard operations on the DP like column creation, sorting, filtering, etc but also includes other operations e.g. Active(column) where an existing column in the DP tells us what rows are active.



In practice:
    (Most Operators are Viewers e.g. Match has the list of columns that can be matched against)
    (There are Views that contain no Operators though)


Backend:

    Interface:
        Collection[BlockGroup]
        InterfaceState

    InterfaceState:
        base_datapanel_id (the datapanel_id of the DP that is the base for the Interface)
        datapanel_id (the datapanel_id of the DP that is the current state of the Interface)
        lineage: List[Operator] (the list of Operators that have been applied to the base DP to get to the current DP)

        assert base_datapanel_id + lineage == datapanel_id (guarantees that the current DP is the result of applying the lineage to the base DP)

        to_code() (convert the list of Operators to a code snippet such that code(base_datapanel_id) == datapanel_id)

        push(Operator) (run an Operator and add it to the lineage)
        pop() (remove the last Operator from the lineage and undo its effect)

    BlockGroup:
        Collection[Block]

    Block:
        List[Operator]
        List[Viewer]
        static: bool (whether the Block is static)

    Operator:
        name: str (the name of the Operator)
        args: List[str] (the list of arguments to the Operator)


    Viewer:



Frontend:
    blocks/
        views/
            <!-- Pure views, no operators -->
            Gallery.svelte
            Table.svelte
        <!-- Operators, which  -->
        Match.svelte
    

<Interface
>
    <Block datapanel_id>
        <Table> // Get the datapanel_id and base_datapanel_id using getContext()
    </Block>
</Interface>