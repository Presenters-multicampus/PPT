from django import forms
from .models import FileModel

class FileForm(forms.ModelForm):
    class Meta:
        model = FileModel
        fields = ('title', 'file')

        def __init__(self, *args, **kwargs):
            super(PostForm, self).__init__(*args, **kwargs)
            self.fields['file'].required = False