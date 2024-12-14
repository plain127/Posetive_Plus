import { create } from "zustand";

const useStore = create(
    (set) => ({
        activeMenu: false,
        setActiveMenu: () => set((state) => ({ activeMenu: !state.activeMenu})),
        activeTooltip: false,
        setActiveTooltip: () => set((state) => ({ activeTooltip: !state.activeTooltip})),
        myPageList: [],
        setMyPageList: (list) => set({myPageList:list}),
        inSubscribed: false,
        setInSubscribed: (boolean) => set({inSubscribed:boolean}),
    })
)

export default useStore;