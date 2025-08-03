# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# User specific environment
if ! [[ "$PATH" =~ "$HOME/.local/bin:$HOME/bin:" ]]
then
    PATH="$HOME/.local/bin:$HOME/bin:$PATH"
fi
export PATH

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# User specific aliases and functions
if [ -d ~/.bashrc.d ]; then
	for rc in ~/.bashrc.d/*; do
		if [ -f "$rc" ]; then
			. "$rc"
		fi
	done
fi

unset rc

# Start Alias
alias rs='squeue --format="%.8i %.8j %.8u %.2t %.8M %.4D %.12b %.12R %.8c"'
alias ls='ls -lFG $LS_OPTIONS'
alias rm='rm -i'
alias mv='mv -i'
alias cp='cp -i'
alias pb='pbcopy'
alias l='less'
alias la='ls -a -lFG $LS_OPTIONS'
alias zonbi='bash ~/../shareP12/scancel_hatakeyama.sh gpu84 gpu85 gpu86'

# End of Alias



# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/appli/miniconda3/24.7.1-py311/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh" ]; then
        . "/home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh"
    else
        export PATH="/home/appli/miniconda3/24.7.1-py311/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

